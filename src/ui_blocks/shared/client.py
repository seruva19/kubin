import random
import string

css_styles = """
html {overflow-y: scroll;}
body {opacity: 0; visibility: hidden;}
body.is-ready {opacity: 1; visibility: visible; transition: opacity 0.5s;}
.block.block-info .min {min-height: initial;}
.block.full-height {height: initial !important;}
.block.block-options {display: block}
.block.block-options span.sr-only + div.wrap {display: block;}
.block.block-options span.sr-only + div.wrap {display: block;}
.block.block-options span.sr-only + div.wrap label {margin: 0 0 5px 0;}
.ui-tabs.left>.tab-nav {position: fixed; left: 0; width: 0;}
.ui-tabs.left>.tab-nav button {min-width: 150px;text-align: left;border: none;}
.ui-tabs.left>.tab-nav button.selected {font-weight: bold;}
.ui-tabs.left>.tabitem {border: none;}
"""

session_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))


def js_loader(resources, params):
    return """
    () => {{
      window._kubinResources = {resources}
      window._kubinParams = {params}
      window._kubinSessionId = '{session_id}'

      const script = document.createElement('script')
      script.src = '/file=client/ui_utils.js?{session_id}'
      script.async = false

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    }}""".format(
        session_id=session_id, resources=resources, params=params
    )
