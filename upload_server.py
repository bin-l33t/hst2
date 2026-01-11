#!/usr/bin/env python3
"""
Simple file upload server for the rectifier project.

Run with: python3 upload_server.py
Then visit: http://localhost:8080

Files are uploaded to the rectifier directory.
"""

import os
import http.server
import socketserver
import cgi
from pathlib import Path
from string import Template

UPLOAD_DIR = Path(__file__).parent.absolute()
PORT = 8080

# Using Template instead of .format() to avoid CSS brace issues
HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
    <title>Rectifier File Upload</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; }
        h2 { color: #7b68ee; margin-top: 40px; }
        .upload-form {
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
        }
        input[type="file"] {
            background: #0f3460;
            color: #fff;
            padding: 15px;
            border: 2px dashed #00d4ff;
            border-radius: 8px;
            width: 100%;
            margin: 10px 0;
            cursor: pointer;
        }
        input[type="file"]:hover {
            border-color: #7b68ee;
        }
        button {
            background: linear-gradient(135deg, #00d4ff, #7b68ee);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 15px;
        }
        button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
        }
        .file-list {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }
        .file-item {
            padding: 10px;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
        }
        .file-item:last-child { border-bottom: none; }
        .file-name { color: #00d4ff; }
        .file-size { color: #888; }
        .success {
            background: #0a4d0a;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .error {
            background: #4d0a0a;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        code {
            background: #0f3460;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
        }
        pre {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>Rectifier Project - File Upload</h1>

    $message

    <div class="upload-form">
        <form action="/upload" method="POST" enctype="multipart/form-data">
            <label for="file">Select a file to upload:</label>
            <input type="file" name="file" id="file" required>
            <br>
            <button type="submit">Upload File</button>
        </form>
    </div>

    <p>Files will be saved to: <code>$upload_dir</code></p>

    <h2>Current Files</h2>
    <div class="file-list">
        $file_list
    </div>

    <h2>Quick Commands</h2>
    <p>Convert uploaded PDF to text:</p>
    <pre><code>pdftotext filename.pdf filename.txt</code></pre>
</body>
</html>
""")

def format_size(size):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def get_file_list():
    """Get HTML list of files in upload directory."""
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            size = format_size(f.stat().st_size)
            files.append(f'<div class="file-item"><span class="file-name">{f.name}</span><span class="file-size">{size}</span></div>')
    return '\n'.join(files) if files else '<div class="file-item">No files yet</div>'

def render_page(message=""):
    """Render the HTML page with current file list."""
    return HTML_TEMPLATE.substitute(
        upload_dir=UPLOAD_DIR,
        file_list=get_file_list(),
        message=message
    )

class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(render_page().encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/upload':
            content_type = self.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                self.send_error(400, 'Expected multipart form data')
                return

            # Parse the form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )

            if 'file' not in form:
                self.send_error(400, 'No file field in form')
                return

            file_item = form['file']
            if not file_item.filename:
                self.send_error(400, 'No file selected')
                return

            # Save the file
            filename = os.path.basename(file_item.filename)
            filepath = UPLOAD_DIR / filename

            with open(filepath, 'wb') as f:
                f.write(file_item.file.read())

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            msg = f'<div class="success">Successfully uploaded: <strong>{filename}</strong> ({format_size(filepath.stat().st_size)})</div>'
            self.wfile.write(render_page(msg).encode())
        else:
            self.send_error(404, 'Not found')

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == '__main__':
    with ReusableTCPServer(("", PORT), UploadHandler) as httpd:
        print(f"Upload server running at http://localhost:{PORT}")
        print(f"Files will be saved to: {UPLOAD_DIR}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
