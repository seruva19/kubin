(global => {
  const kubin = global.kubin = {
    client_prefix : '/file=',
    client_id: String(Date.now().toString(32) + Math.random().toString(16)).replace(/\./g, ''),
    utils: {}
  }

  const randomHash = () => {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let randomHash = '';
    for (let i = 0; i < length; i++) {
      const randomIndex = Math.floor(Math.random() * characters.length);
      randomHash += characters.charAt(randomIndex);
    }

    return randomHash;
  }

  const loadJs = kubin.utils.loadJs = async url => {
    return new Promise(resolve => { 
      const script = document.createElement('script')
      script.src = `${url}?${randomHash()}`
      script.async = false
      script.onload = resolve

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    })
  }

  const loadCss = kubin.utils.loadCss = url => {
    const style = document.createElement('link')
    style.href = `${url}?${randomHash()}`
    style.rel = 'stylesheet'
    
    const head = document.getElementsByTagName("head")[0]
    head.appendChild(style)
  }

  kubin.notify = {
    lib: undefined,
    success: message => kubin.notify.lib.success(message),
    warning: message => kubin.notify.lib.warning(message),
    error: message => kubin.notify.lib.error(message)
  }
  
  const extensionResources = window._kubinResources ?? []

  Promise.all([
    loadCss(`${kubin.client_prefix}client/ui_utils.css`),
    loadJs(`${kubin.client_prefix}client/3party/notyf.min.js`),
    loadCss(`${kubin.client_prefix}client/3party/notyf.min.css`),
  ].concat(extensionResources.map(resource => {
    const resourceUrl = `${kubin.client_prefix}${resource}`
    if (resource.endsWith('.css')) {
      return loadCss(resourceUrl)
    } else if (resource.endsWith('.js')) {
      return loadJs(resourceUrl)
    } else {
      return Promise.resolve()
    }
  })))
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