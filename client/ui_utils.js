(global => {
  const kubin = global.kubin = {
    client_prefix: '/file=',
    client_id: String(Date.now().toString(32) + Math.random().toString(16)).replace(/\./g, ''),
    utils: {},
    options: {}
  }

  const randomString = () => {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    let randomHash = ''
    let length = 8

    for (let i = 0; i < length; i++) {
      const randomIndex = Math.floor(Math.random() * characters.length)
      randomHash += characters.charAt(randomIndex)
    }

    return randomHash
  }

  const loadJs = kubin.utils.loadJs = async url => {
    return new Promise(resolve => {
      const script = document.createElement('script')
      script.src = `${url}?${randomString()}`
      script.async = false
      script.onload = resolve

      const head = document.getElementsByTagName("head")[0]
      head.appendChild(script)
    })
  }

  const loadCss = kubin.utils.loadCss = url => {
    const style = document.createElement('link')
    style.href = `${url}?${randomString()}`
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
    customEventListeners.init()

    console.log('UI successfully loaded')
    kubin.UI.reveal()
  })

  const customEventListeners = {
    init: () => {
      window.document.addEventListener('click', e => {
        if (e.target.classList.contains('options-select')) {
          Array.from(document.querySelectorAll('.options-select')).forEach(option => {
            option.classList.remove('selected')
          })

          e.target.classList.add('selected')
          const id = e.target.id

          Array.from(document.querySelectorAll('.options-block')).forEach(option => {
            option.classList.remove('active')
          })

          document.querySelector(`.${id}`).classList.add('active')
        } else if (e.target.nextElementSibling?.classList.contains('thumbnails')) {
          const targetGallery = e.target.nextElementSibling
          const thumbnailsSelector = `.gallery-active button img`
          targetGallery.classList.add('gallery-active')

          let position = 0
          const allImages = Array.from(e.target.nextElementSibling.querySelectorAll(thumbnailsSelector))
          allImages.forEach((image, index) => {
            if (image.src === e.target.src) {
              position = index - 1 == -1 ? allImages.length - 1 : index - 1
            }
          })

          window._kubinGalleryThumbnails = allImages
          window._kubinThumbnailIndex = position

          const gallery = window._kubinGallery = new SimpleLightbox(thumbnailsSelector, {
            sourceAttr: 'src',
            download: 'Download image',
            animationSlide: false,
            swipeTolerance: Number.MAX_VALUE
          })

          gallery.on('closed.simplelightbox', () => {
            targetGallery.classList.remove('gallery-active')
            window._kubinGallery = undefined
            gallery.destroy()
          })

          gallery.openPosition(position)
          window._kubinGalleryThumbnails[position].click()
        } else if (e.target.parentNode?.classList.contains('sl-image')) {
          window._kubinGallery?.next()

          window._kubinThumbnailIndex++
          if (window._kubinThumbnailIndex == window.window._kubinGalleryThumbnails.length) {
            window._kubinThumbnailIndex = 0
          }

          window._kubinGalleryThumbnails[window._kubinThumbnailIndex]?.click()
        } else if (e.target.classList.contains('sl-prev')) {
          window._kubinThumbnailIndex--
          if (window._kubinThumbnailIndex == -1) {
            window._kubinThumbnailIndex = window.window._kubinGalleryThumbnails.length - 1
          }

          window._kubinGalleryThumbnails[window._kubinThumbnailIndex].click()
        } else if (e.target.classList.contains('sl-next')) {
          window._kubinThumbnailIndex++
          if (window._kubinThumbnailIndex == window.window._kubinGalleryThumbnails.length) {
            window._kubinThumbnailIndex = 0
          }

          window._kubinGalleryThumbnails[window._kubinThumbnailIndex].click()
        }
      })
    }
  }

  kubin.UI = {
    loadParams: params => {
      const panelResize = params['ui.allow_params_panel_resize']
      if (undefined !== panelResize) {
        kubin.UI.resizablePanels(panelResize)
      }

      const verticalAlignment = params['ui.enable_vertical_alignment']
      if (undefined !== verticalAlignment) {
        kubin.UI.verticalAlignment(verticalAlignment)
      }

      const fullScreenPanel = params['ui.full_screen_panel']
      if (undefined !== fullScreenPanel) {
        kubin.UI.fullScreenUI(fullScreenPanel)
      }

      const pipeline = window._kubinParams['general.pipeline']
      const model_name = window._kubinParams['general.model_name']

      document.querySelectorAll('body[class*="pipeline-"]').forEach(b => {
        for (let i = b.classList.length - 1; i >= 0; i--) {
          const className = b.classList[i]
          if (className.startsWith('pipeline-')) {
            b.classList.remove(className)
          }
        }
      })

      document.body.classList.add(`pipeline-${pipeline}-${model_name}`)

      if (model_name == 'kd20' && pipeline == 'diffusers') {
        kubin.notify.error('You cannot use a 2.0 model with the diffusers pipeline! Native pipeline will be used')
      } else if (model_name == 'kd22' && pipeline == 'native') {
        kubin.notify.error('You cannot use a 2.2 model with the native pipeline! Diffusers pipeline will be used')
      }
    },

    resizablePanels: panelResize => {
      panelResize && Array.from(document.getElementsByClassName('block-params')).forEach(paramsBlock => {
        const anchor = document.createElement('div')
        const randomClass = `block-resizable-anchor-${randomString()}`
        anchor.className = `block-resizable-anchor ${randomClass}`
        anchor.title = 'Drag to resize main panel'

        const pb = paramsBlock
        pb.appendChild(anchor)

        let resizing = false
        let initialWidth
        let startX

        window.document.addEventListener('mousedown', e => {
          if (e.target.classList.contains(randomClass)) {
            anchor.classList.add('activated')
            document.body.classList.add('unselectable')
            initialWidth = pb.clientWidth
            resizing = true
            startX = e.clientX
          }
        })

        window.document.addEventListener('mousemove', e => {
          if (resizing) {
            const distanceX = e.clientX - startX
            pb.style.maxWidth = `${initialWidth + distanceX}px`
          }
        })

        window.document.addEventListener('mouseup', e => {
          anchor.classList.remove('activated')
          document.body.classList.remove('unselectable')
          resizing = false
        })
      })

      !panelResize && Array.from(document.getElementsByClassName('block-resizable-anchor')).forEach(anchor => {
        anchor.remove
      })
    },

    verticalAlignment: verticalAlignment => {
      Array.from(document.getElementsByClassName('block-params')).forEach(paramsBlock => {
        verticalAlignment && paramsBlock.parentElement.classList.add('block-params-vertical-alignment')
        !verticalAlignment && paramsBlock.parentElement.classList.remove('block-params-vertical-alignment')
      })
    },

    fullScreenUI: fullScreenUI => {
      fullScreenUI && document.body.classList.add('gradio-full')
      !fullScreenUI && document.body.classList.remove('gradio-full')
    },

    reveal: () => {
      document.getElementsByTagName('html')[0].classList.add('is-ready')
    }
  }

  let changes = {}
  kubin.utils.optionsChanged = (key, value, requiresRestart) => {
    changes[key] = { value, requiresRestart }
    window._kubinParams[key] = value
  }

  kubin.utils.changedOptions = () => {
    const serverChanges = {}
    Object.keys(changes).forEach(key => {
      serverChanges[key] = changes[key].value
    })

    return JSON.stringify(serverChanges)
  }

  kubin.utils.processOptionsChanges = (success, _) => {
    const changeInfo = document.getElementById('options-info')

    if (success) {
      kubin.notify.success('Changes successfully applied')
      kubin.UI.loadParams(JSON.parse(kubin.utils.changedOptions()))

      changeInfo.innerHTML = (cs => {
        return cs.map(key => {
          const { value, requiresRestart } = changes[key]
          return `
            <span style='font-weight: bold'>${key}</span>
            <span>was successfully changed to</span>
            <span style='font-weight: bold'>${value}</span>
            <span style='color: red'>${requiresRestart ? ' restart required' : ''}</span>
          `
        }).join('<br>')
      })(Object.keys(changes))

      changes = {}
    } else {
      kubin.notify.error('Failed to save changes!')
      changeInfo.innerHTML = 'Failed to save changes!'
    }

    return [success]
  }
})(window)
