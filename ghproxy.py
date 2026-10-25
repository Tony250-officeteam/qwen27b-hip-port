"""Minimal CONNECT proxy that pins github.com (and friends) to real IPs.

The local DNS resolver returns 198.18.0.16 for github.com, which black-holes the
connection. Everything else reaches GitHub fine. So instead of editing the hosts
file (admin rights), route the CONNECT tunnel to a known-good IP.

usage:  python ghproxy.py [port]
then:   set HTTPS_PROXY=http://127.0.0.1:<port>
"""
import select
import socket
import socketserver
import sys
import threading

# github.com is poisoned locally; api.github.com and raw.githubusercontent.com
# resolve correctly, so they are only pinned as a safety net.
PIN = {
    "github.com": ["20.205.243.166", "140.82.112.3", "140.82.113.3"],
    "api.github.com": ["20.205.243.168", "140.82.113.6"],
    "raw.githubusercontent.com": ["185.199.108.133", "185.199.109.133"],
    "codeload.github.com": ["140.82.112.9", "140.82.113.9"],
    "objects.githubusercontent.com": ["185.199.108.133"],
}

BUFSIZE = 65536


def connect_pinned(host, port):
    """Return a socket to host:port, using a pinned IP when we have one."""
    last = None
    for ip in PIN.get(host, []):
        try:
            s = socket.create_connection((ip, port), timeout=15)
            return s, ip
        except Exception as e:  # noqa: BLE001 - report the last error
            last = e
    if host in PIN:
        raise last if last else OSError(f"no usable IP for {host}")
    return socket.create_connection((host, port), timeout=15), host


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        head = b""
        self.request.settimeout(30)
        try:
            while b"\r\n\r\n" not in head:
                chunk = self.request.recv(4096)
                if not chunk:
                    return
                head += chunk
        except socket.timeout:
            return

        first = head.split(b"\r\n", 1)[0].decode("latin1", "replace")
        parts = first.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            self.request.sendall(b"HTTP/1.1 405 Only CONNECT\n\n")
            return

        target = parts[1]
        host, _, port_s = target.rpartition(":")
        port = int(port_s or "443")

        try:
            upstream, used = connect_pinned(host, port)
        except Exception as e:  # noqa: BLE001
            self.request.sendall(b"HTTP/1.1 502 Cannot reach upstream\n\n")
            print(f"  FAIL {target}: {e}", flush=True)
            return

        self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        print(f"  TUNNEL {target} -> {used}", flush=True)

        socks = [self.request, upstream]
        try:
            while True:
                r, _, x = select.select(socks, [], socks, 60)
                if x or not r:
                    break
                for s in r:
                    data = s.recv(BUFSIZE)
                    if not data:
                        return
                    (upstream if s is self.request else self.request).sendall(data)
        except Exception:  # noqa: BLE001 - client went away
            pass
        finally:
            try:
                upstream.close()
            except Exception:  # noqa: BLE001
                pass


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    srv = Server(("127.0.0.1", port), Handler)
    print(f"listening on 127.0.0.1:{port}", flush=True)
    print("set HTTPS_PROXY=http://127.0.0.1:%d" % port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
