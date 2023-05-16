css_styles = '''
html {overflow-y: scroll;}
.block.block-info .min {min-height: initial;}
.block.full-height {height: initial !important;}
'''

js_scripts = '''
() => {
  const script = document.createElement('script')
  script.src = '/file=client/ui_utils.js'
  script.async = false

  const head = document.getElementsByTagName("head")[0]
  head.appendChild(script)
}
'''