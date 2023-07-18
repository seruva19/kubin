import os


def minify_and_bundle():
    client_root = "./client"

    css = ""
    js = ""

    for file in os.listdir(f"{client_root}"):
        if file.endswith(".css"):
            css += open(f"{client_root}/{file}", "r").read()
            css += "\n"
        if file.endswith(".js"):
            js += open(f"{client_root}/{file}", "r").read()
            js += ";\n"

    with open(f"{client_root}/dist/bundle.css", "w") as f:
        f.write(css)

    with open(f"{client_root}/dist/bundle.js", "w") as f:
        f.write(js)


minify_and_bundle()
