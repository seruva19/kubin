(global => {
  const kubin = {
    client_prefix: '/file=',
    client_id: String(Date.now().toString(32) + Math.random().toString(16)).replace(/\./g, ''),
    utils: {},
    options: {},
    UI: {},
    notify: {},
    randomString: () => {
      const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
      let randomHash = ''
      let length = 8

      for (let i = 0; i < length; i++) {
        const randomIndex = Math.floor(Math.random() * characters.length)
        randomHash += characters.charAt(randomIndex)
      }

      return randomHash
    }
  }

  const loadJs = kubin.utils.loadJs = async url => {
    return new Promise(resolve => {
      const script = document.createElement('script')
      script.src = `${url}?${kubin.randomString()}`
      script.async = false
      script.onload = resolve

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    })
  }

  const loadCss = kubin.utils.loadCss = url => {
    const style = document.createElement('link')
    style.href = `${url}?${kubin.randomString()}`
    style.rel = 'stylesheet'

    const head = document.getElementsByTagName("head")[0]
    head.appendChild(style)
  }

  kubin.notify = {
    lib: undefined,
    success: message => kubin.notify.lib.success(message),
    warning: message => kubin.notify.lib.open({ type: 'warning', message }),
    error: message => kubin.notify.lib.error(message)
  }

  const extensionResources = window._kubinResources ?? []

  Promise.all([
    loadCss(`${kubin.client_prefix}client/dist/bundle.css`),
    loadJs(`${kubin.client_prefix}client/3party/notyf.min.js`),
    loadCss(`${kubin.client_prefix}client/3party/notyf.min.css`),
    loadJs(`${kubin.client_prefix}client/3party/simple-lightbox.min.js`),
    loadCss(`${kubin.client_prefix}client/3party/simple-lightbox.min.css`),
  ].concat(extensionResources.map(resource => {
    const resourceUrl = `${kubin.client_prefix}${resource}`

    if (resource.endsWith('.css')) {
      return loadCss(resourceUrl)
    } else if (resource.endsWith('.js')) {
      return loadJs(resourceUrl)
    } else {
      return Promise.resolve()
    }
  }))).then(() => {
    kubin.notify.lib = new Notyf({
      duration: 5000,
      ripple: false,
      position: { x: 'right', y: 'bottom' },
      types: [
        { type: 'warning', background: 'orange', icon: false },
        { type: 'success', background: 'seagreen', icon: false },
        { type: 'error', background: 'indianred', icon: false }
      ]
    })

    kubin.UI.loadParams(window._kubinParams)
    kubin.UI.customEventListeners()
    kubin.UI.defaultConditions()

    kubin.UI.reveal()
    console.log('UI successfully loaded')
  })

  global.kubin = kubin
})(window)
