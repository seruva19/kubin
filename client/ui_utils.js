(global => {
  const kubin = global.kubin = {
    client_root : '/file=client/',
    utils: {}
  }

  const loadJs = kubin.utils.loadJs = async url => {
    return new Promise(resolve => { 
      const script = document.createElement('script')
      script.src = url
      script.async = false
      script.onload = resolve

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    })
  }

  const loadCss = kubin.utils.loadCss = url => {
    const style = document.createElement('link')
    style.href = url
    style.rel = 'stylesheet'
    
    const head = document.getElementsByTagName("head")[0]
    head.appendChild(style)
  }

  kubin.notify = {
    lib: undefined,
    success: message => kubin.notify.lib.success(message)
  }
  
  Promise.all([
    loadCss(kubin.client_root + 'ui_utils.css'),
    loadJs(kubin.client_root + '3party/notyf.min.js'),
    loadCss(kubin.client_root + '3party/notyf.min.css'),
  ])
  .then(() => {
    kubin.notify.lib = new Notyf({
      duration: 5000,
      ripple: false,
      position: {x: 'right', y: 'bottom'},
      types: [
        {type: 'warning', background: 'orange', icon: false},
        {type: 'success', background: 'seagreen', icon: false},
        {type: 'error', background: 'indianred', icon: false}
      ]
    })

    kubin.notify.success('kubin client library: loaded')
    console.log('kubin client library: loaded')
  })
})(window)