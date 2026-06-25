"""GitHub REST API client for source-code grounding.

Wraps two endpoints:
- GET /search/code?q=...  -> ranked list of matching files across public repos
- GET /repos/{owner}/{repo}/contents/{path}  -> raw file content (base64-decoded)

The point is to give the research pipeline access to the actual implementation
behind a concept (Linux kernel sources, K8s/Envoy/Cilium, FRR, libbpf, etc.)
so research_brief.md can cite concrete code paths instead of just web docs.

Public endpoints work with any valid PAT. Fine-grained tokens without `repo`
scope are fine for the public read paths we use here.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any

import requests


# ---- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class CodeMatch:
    repo_full_name: str
    path: str
    sha: str
    html_url: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class FileContents:
    repo_full_name: str
    path: str
    sha: str
    size: int
    content: str
    html_url: str | None = None


@dataclass(frozen=True)
class RateLimit:
    remaining: int | None
    reset_at: int | None  # unix timestamp


# ---- Topic -> repo map ----------------------------------------------------
#
# Naive code search (e.g. `"TCP SYN backlog" in:file language:c`) ranks by
# keyword relevance and returns tangentially related files (mongoose.c, an
# obscure embedded TCP stack) instead of the Linux kernel sources we actually
# want. This curated map fixes that for the channel's core topics.
#
# IMPORTANT: keys are matched in **insertion order**, first match wins. More
# specific keys (e.g. `ipv6`, `tcp_syn_backlog`) MUST come before generic
# ones (e.g. `ip`, `tcp`, `socket`) so the matcher picks the most specific
# entry for ambiguous topics like "IPv6 socket".

TOPIC_REPO_MAP: dict[str, list[tuple[str, str]]] = {
    # ---- TCP / Linux networking ----------------------------------------------
    "tcp_syn_backlog": [
        ("torvalds/linux", "net/ipv4/tcp_input.c"),
        ("torvalds/linux", "net/ipv4/tcp_minisocks.c"),
        ("torvalds/linux", "net/ipv4/tcp_fastopen.c"),
        ("torvalds/linux", "include/net/tcp.h"),
    ],
    "three_way_handshake": [
        ("torvalds/linux", "net/ipv4/tcp_input.c"),
        ("torvalds/linux", "net/ipv4/tcp_minisocks.c"),
    ],
    "ipv6": [
        ("torvalds/linux", "net/ipv6/tcp_ipv6.c"),
        ("torvalds/linux", "net/ipv6/inet6_connection_sock.c"),
        ("torvalds/linux", "net/ipv6/addrconf.c"),
    ],
    "ipv4": [
        ("torvalds/linux", "net/ipv4/tcp_ipv4.c"),
        ("torvalds/linux", "net/ipv4/af_inet.c"),
        ("torvalds/linux", "net/ipv4/ip_input.c"),
    ],
    "ip": [
        ("torvalds/linux", "net/ipv4/route.c"),
        ("torvalds/linux", "net/ipv4/fib_semantics.c"),
    ],
    "tcp": [
        ("torvalds/linux", "net/ipv4/tcp_input.c"),
        ("torvalds/linux", "net/ipv4/tcp_output.c"),
        ("torvalds/linux", "net/ipv4/tcp_timer.c"),
        ("torvalds/linux", "net/ipv4/tcp_cong.c"),
        ("torvalds/linux", "include/net/tcp.h"),
    ],
    "syn": [
        ("torvalds/linux", "net/ipv4/tcp_input.c"),
        ("torvalds/linux", "net/ipv4/tcp_minisocks.c"),
    ],
    "socket": [
        ("torvalds/linux", "net/socket.c"),
        ("torvalds/linux", "net/core/sock.c"),
    ],
    "epoll": [
        ("torvalds/linux", "fs/eventpoll.c"),
    ],
    "iptables": [
        ("torvalds/linux", "net/ipv4/netfilter/ip_tables.c"),
        ("torvalds/linux", "net/netfilter/x_tables.c"),
    ],
    "nftables": [
        ("torvalds/linux", "net/netfilter/nf_tables_core.c"),
        ("torvalds/linux", "net/netfilter/nf_tables_api.c"),
    ],
    "conntrack": [
        ("torvalds/linux", "net/netfilter/nf_conntrack_core.c"),
    ],
    "nat": [
        ("torvalds/linux", "net/netfilter/nf_nat_core.c"),
    ],
    # ---- Routing (FRRouting) -----------------------------------------------
    "bgp": [
        ("FRRouting/frr", "bgpd/bgp_attr.c"),
        ("FRRouting/frr", "bgpd/bgp_route.c"),
        ("FRRouting/frr", "bgpd/bgp_packet.c"),
        ("FRRouting/frr", "bgpd/bgp_zebra.c"),
    ],
    "ospf": [
        ("FRRouting/frr", "ospfd/ospf_spf.c"),
        ("FRRouting/frr", "ospfd/ospf_neighbor.c"),
    ],
    "isis": [
        ("FRRouting/frr", "isisd/isis_spf.c"),
    ],
    # ---- eBPF / observability ----------------------------------------------
    "ebpf": [
        ("libbpf/libbpf", "src/libbpf.c"),
        ("libbpf/libbpf", "src/bpf.c"),
        ("torvalds/linux", "kernel/bpf/syscall.c"),
        ("torvalds/linux", "kernel/bpf/core.c"),
    ],
    "cilium": [
        ("cilium/cilium", "pkg/datapath/linux/conntrack.go"),
        ("cilium/cilium", "pkg/datapath/loader.go"),
    ],
    # ---- Kubernetes / containers -------------------------------------------
    "kubernetes": [
        ("kubernetes/kubernetes", "pkg/kubelet/kubelet.go"),
        ("kubernetes/kubernetes", "pkg/proxy/ipvs/proxier.go"),
    ],
    "k8s": [
        ("kubernetes/kubernetes", "pkg/kubelet/kubelet.go"),
        ("kubernetes/kubernetes", "pkg/proxy/ipvs/proxier.go"),
    ],
    "cni": [
        ("containernetworking/cni", "libcni/api.go"),
        ("containernetworking/plugins", "plugins/main/bridge/bridge.go"),
    ],
    # ---- Proxies / service mesh --------------------------------------------
    "envoy": [
        ("envoyproxy/envoy", "source/common/network/connection_impl.cc"),
        ("envoyproxy/envoy", "source/common/upstream/cluster_manager_impl.cc"),
    ],
}


def match_topic_files(topic: str) -> list[tuple[str, str]]:
    """Return the curated (repo, path) list for `topic`, or [] if no match.

    Matching is case-insensitive substring search over TOPIC_REPO_MAP keys,
    in insertion order. First match wins, so callers must order the map
    most-specific-first.
    """
    if not topic:
        return []
    needle = topic.lower()
    for key, files in TOPIC_REPO_MAP.items():
        if key in needle:
            return list(files)
    return []


# ---- Errors ---------------------------------------------------------------


class GitHubClientError(RuntimeError):
    """Raised on any GitHub API failure (HTTP, parse, or rate-limit)."""


# ---- Client ---------------------------------------------------------------


class GitHubClient:
    """Thin client for the GitHub REST API v3."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        timeout_seconds: float = 30.0,
        user_agent: str = "knowledge-pipeline-mvp1",
    ) -> None:
        if not token:
            raise ValueError("GitHub token must not be empty")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._user_agent = user_agent
        self._last_rate_limit: RateLimit | None = None

    @property
    def last_rate_limit(self) -> RateLimit | None:
        return self._last_rate_limit

    # ---- Transport --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self._user_agent,
        }

    def _record_rate_limit(self, response: requests.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        try:
            remaining_int: int | None = int(remaining) if remaining is not None else None
        except ValueError:
            remaining_int = None
        try:
            reset_int: int | None = int(reset) if reset is not None else None
        except ValueError:
            reset_int = None
        self._last_rate_limit = RateLimit(remaining=remaining_int, reset_at=reset_int)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise GitHubClientError(f"HTTP error calling {url}: {exc}") from exc
        self._record_rate_limit(response)
        if response.status_code == 403 and self._last_rate_limit and self._last_rate_limit.remaining == 0:
            raise GitHubClientError(
                f"GitHub rate limit exhausted (resets at unix {self._last_rate_limit.reset_at})"
            )
        if response.status_code >= 400:
            # Try to surface the GitHub error message.
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise GitHubClientError(
                f"GitHub API {response.status_code} on {url}: {message}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubClientError(f"Non-JSON response from {url}: {exc}") from exc

    # ---- Public API -------------------------------------------------------

    def search_code(self, query: str, max_results: int = 10) -> list[CodeMatch]:
        """Search code across public GitHub repositories.

        Args:
            query: a GitHub code-search expression. Supports repo:, path:,
                language:, in:file qualifiers. See
                https://docs.github.com/en/search-code-search/understanding-the-search-syntax
            max_results: 1..100; the API caps per_page at 100.

        Returns:
            Ranked list of CodeMatch (GitHub's relevance order).

        Raises:
            ValueError: on invalid max_results.
            GitHubClientError: on HTTP, parse, or rate-limit failures.
        """
        if max_results < 1 or max_results > 100:
            raise ValueError("max_results must be between 1 and 100")
        if not query.strip():
            raise ValueError("query must not be empty")

        matches: list[CodeMatch] = []
        page = 1
        per_page = min(max_results, 100)
        while len(matches) < max_results:
            body = self._get(
                "/search/code",
                params={"q": query, "per_page": per_page, "page": page},
            )
            items = body.get("items") or []
            if not items:
                break
            for item in items:
                if len(matches) >= max_results:
                    break
                matches.append(_parse_code_match(item))
            total_so_far = page * per_page
            total_count = int(body.get("total_count") or 0)
            if total_so_far >= total_count:
                break
            page += 1
        return matches

    def get_file_contents(
        self,
        repo_full_name: str,
        path: str,
        ref: str | None = None,
    ) -> FileContents:
        """Fetch the raw content of a single file from a public repo.

        Args:
            repo_full_name: e.g. "torvalds/linux".
            path: repo-relative file path, e.g. "net/ipv4/tcp_input.c".
            ref: optional branch / tag / sha. Defaults to the repo's default branch.

        Raises:
            GitHubClientError: on HTTP, parse, or rate-limit failures.
        """
        if not repo_full_name or "/" not in repo_full_name:
            raise ValueError("repo_full_name must be 'owner/repo'")
        if not path:
            raise ValueError("path must not be empty")

        params: dict[str, Any] = {"ref": ref} if ref else None
        body = self._get(f"/repos/{repo_full_name}/contents/{path}", params=params)
        return _parse_file_contents(repo_full_name, body)

    def find_topic_sources(
        self,
        topic: str,
        *,
        max_files: int = 8,
        language: str | None = "c",
    ) -> list[FileContents]:
        """Return source files relevant to `topic`.

        Two-tier strategy:
          1. If the topic (case-insensitive substring) matches a key in
             TOPIC_REPO_MAP, fetch those curated files directly. This gives
             real Linux-kernel / FRR / K8s / Cilium / Envoy sources for the
             channel's core internals topics, where naive code search would
             return unrelated embedded stacks.
          2. Fall back to GitHub code search with the query
             `"{topic} in:file language:{language}"`.

        Returns up to max_files FileContents, skipping any that fail to fetch
        (binary, too large, deleted, etc.).
        """
        if max_files < 1:
            raise ValueError("max_files must be >= 1")

        # Tier 1: curated topic -> repo map
        mapped = match_topic_files(topic)[:max_files]
        if mapped:
            results: list[FileContents] = []
            for repo, path in mapped:
                try:
                    results.append(self.get_file_contents(repo, path))
                except GitHubClientError:
                    continue
            return results

        # Tier 2: generic code search
        query_parts = [topic.strip(), "in:file"]
        if language:
            query_parts.append(f"language:{language}")
        query = " ".join(p for p in query_parts if p)
        matches = self.search_code(query, max_results=max_files)
        results = []
        for match in matches:
            try:
                results.append(
                    self.get_file_contents(match.repo_full_name, match.path)
                )
            except GitHubClientError:
                continue
        return results


# ---- Response parsing -----------------------------------------------------


def _parse_code_match(item: dict[str, Any]) -> CodeMatch:
    repo = item.get("repository") or {}
    return CodeMatch(
        repo_full_name=str(repo.get("full_name", "")),
        path=str(item.get("path", "")),
        sha=str(item.get("sha", "")),
        html_url=item.get("html_url"),
        score=item.get("score"),
    )


def _parse_file_contents(repo_full_name: str, body: dict[str, Any]) -> FileContents:
    content_b64 = body.get("content")
    encoding = body.get("encoding", "")
    if not content_b64:
        raise GitHubClientError(
            f"GitHub contents response missing 'content' for {repo_full_name}/{body.get('path', '?')}"
        )
    if encoding != "base64":
        raise GitHubClientError(
            f"Unexpected content encoding {encoding!r} for {repo_full_name}/{body.get('path', '?')}"
        )
    try:
        # GitHub wraps base64 at 60-char lines.
        raw = base64.b64decode(content_b64.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise GitHubClientError(
            f"Failed to base64-decode {repo_full_name}/{body.get('path', '?')}: {exc}"
        ) from exc
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Binary file — return a lossy decoding so the caller can still see SOMETHING.
        decoded = raw.decode("utf-8", errors="replace")

    return FileContents(
        repo_full_name=repo_full_name,
        path=str(body.get("path", "")),
        sha=str(body.get("sha", "")),
        size=int(body.get("size") or len(raw)),
        content=decoded,
        html_url=body.get("html_url"),
    )
