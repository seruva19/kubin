import random
import string

from ui_blocks.shared.compatibility import generate_rules

css_styles = (
    """
html {overflow-y: scroll;}
html:not(.is-ready):before {content: " ";position: fixed;right: 50%;bottom: 50%;
margin: auto; border: 20px solid #EAF0F6; border-radius: 50%; border-top: 20px solid var(--loader-color); width: 20px; height: 20px;animation: loader 2s linear infinite;}
@keyframes loader {0% { transform: rotate(0deg); }100% { transform: rotate(360deg); }}
html:not(.is-ready) body {opacity: 0; visibility: hidden;}
html.is-ready body {opacity: 1; visibility: visible; transition: opacity 0.5s;}
.block.block-info .min {min-height: initial;}
.block.full-height {height: initial !important;}
.block.block-options {display: block}
.block.block-options span.sr-only + div.wrap {display: block;}
.block.block-options span.sr-only + div.wrap {display: block;}
.block.block-options span.sr-only + div.wrap label {margin: 0 0 5px 0;}
.ui-tabs.left>.tab-nav {position: fixed; left: 0; max-width: 200px; height: 100%; overflow-y: scroll; display: inherit; padding-bottom: 20px;}
.ui-tabs.left>.tab-nav button {min-width: 150px;text-align: left;border: none;}
.ui-tabs.left>.tab-nav button.selected {font-weight: bold;}
.ui-tabs.left>.tabitem {border: none;}
"""
    + generate_rules()
)

random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))


def js_loader(resources, params):
    return """
    () => {{
      window._kubinSession = [Date.now(), Math.floor(Math.random() * 10000)].join('-')
      window._kubinResources = {resources}
      window._kubinParams = {params}
      
      const script = document.createElement('script')
      script.src = '/file=client/dist/bundle.js?{random_id}'
      script.async = false

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)

      return [window._kubinSession]
    }}""".format(
        random_id=random_id, resources=resources, params=params
    )
