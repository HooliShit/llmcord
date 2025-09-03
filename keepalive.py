# keepalive.py
import os, http.server, socketserver
PORT = int(os.environ.get("PORT", "10000"))
Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
