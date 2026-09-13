"""A self-signed certificate so the dashboard can be opened over https:// on the local network.

Why this exists: browsers only hand out the microphone in a *secure context*. `localhost` counts as one,
`http://192.168.1.42:8010` does not -- so a phone or a judge's laptop opening the LAN link gets
"microphone API unavailable" and the live call cannot start. A Cloudflare tunnel solves it, but it needs
cloudflared installed and outbound internet, and conference wifi is exactly where both tend to fail.

So the server can serve HTTPS itself. The certificate is self-signed, which means the first visitor sees a
browser warning and has to click through ("Advanced" -> "Proceed"); after that the microphone works. The
certificate lists every LAN address of this machine, so the same file works for every device on the wifi.

Generated with `cryptography` when it is installed, otherwise with the `openssl` binary (Git for Windows
ships one). Returns (cert_path, key_path), or None when neither is available.
"""
from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import shutil
import socket
import subprocess

log = logging.getLogger("detector.tls")


def lan_addresses() -> list[str]:
    """Every non-loopback IPv4 address this machine answers on."""
    out: list[str] = []
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                out.append(ip)
    except OSError:
        pass
    if not out:                      # the hostname lookup misses some VPN/docker setups
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            out.append(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    return sorted(set(out))


def _names() -> tuple[list[str], list[str]]:
    raw = ["localhost", socket.gethostname()]
    try:
        raw.append(socket.getfqdn())
    except OSError:
        pass
    # getfqdn() can come back as several names in one string on a domain-joined machine; a SAN entry
    # containing a space is rejected outright, which would make the whole certificate useless.
    hosts = {tok for item in raw for tok in str(item).split() if tok and " " not in tok}
    ips = ["127.0.0.1"] + lan_addresses()
    return sorted(hosts), sorted(set(ips))


def _with_cryptography(cert: str, key: str, hosts: list[str], ips: list[str]) -> bool:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Altur Voice Shield (self-signed)")])
    san = [x509.DNSName(h) for h in hosts] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = datetime.datetime.now(datetime.timezone.utc)
    crt = (x509.CertificateBuilder()
           .subject_name(name).issuer_name(name).public_key(k.public_key())
           .serial_number(x509.random_serial_number())
           .not_valid_before(now - datetime.timedelta(days=1))
           .not_valid_after(now + datetime.timedelta(days=365))
           .add_extension(x509.SubjectAlternativeName(san), critical=False)
           .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
           .sign(k, hashes.SHA256()))
    with open(key, "wb") as fh:
        fh.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                 serialization.NoEncryption()))
    with open(cert, "wb") as fh:
        fh.write(crt.public_bytes(serialization.Encoding.PEM))
    return True


def _with_openssl(cert: str, key: str, hosts: list[str], ips: list[str]) -> bool:
    exe = shutil.which("openssl")
    if not exe:
        return False
    san = ",".join([f"DNS:{h}" for h in hosts] + [f"IP:{i}" for i in ips])
    cmd = [exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
           "-keyout", key, "-out", cert, "-subj", "/CN=Altur Voice Shield (self-signed)",
           "-addext", f"subjectAltName={san}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("openssl could not make a certificate: %s", exc)
        return False


def ensure_cert(directory: str, regenerate: bool = False) -> tuple[str, str] | None:
    """Return (cert_path, key_path), making them if needed. None when no generator is available."""
    os.makedirs(directory, exist_ok=True)
    cert = os.path.join(directory, "dev-cert.pem")
    key = os.path.join(directory, "dev-key.pem")
    if not regenerate and os.path.exists(cert) and os.path.exists(key):
        return cert, key
    hosts, ips = _names()
    for fn in (_with_cryptography, _with_openssl):
        if fn(cert, key, hosts, ips):
            log.info("self-signed certificate for %s / %s -> %s", ", ".join(hosts), ", ".join(ips), cert)
            return cert, key
    log.warning("no certificate generator found (pip install cryptography, or install openssl)")
    return None
