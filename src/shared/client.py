import random
import string

css_styles = """
html {overflow-y: scroll;}
.block.block-info .min {min-height: initial;}
.block.full-height {height: initial !important;}
"""

random_hash = "".join(random.choices(string.ascii_letters + string.digits, k=8))


def js_loader(resources):
    return """
    () => {{
      window._kubinResources = {resources}

      const script = document.createElement('script')
      script.src = '/file=client/ui_utils.js?{random_hash}'
      script.async = false

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    }}""".format(
        random_hash=random_hash, resources=resources
    )
