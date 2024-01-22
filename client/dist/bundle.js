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
;
(global => {
  kubin.UI.customEventListeners = () => {
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
      } else if (e.target.parentNode?.nextElementSibling?.classList.contains('thumbnails')) {
        const targetGallery = e.target.parentNode.nextElementSibling
        const thumbnailsSelector = `.gallery-active button img`
        targetGallery.classList.add('gallery-active')

        let position = 0
        const allImages = Array.from(targetGallery.querySelectorAll(thumbnailsSelector))
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

  kubin.UI.loadParams = params => {
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

    let pipeline = window._kubinParams['general.pipeline']
    const model_name = window._kubinParams['general.model_name']

    document.querySelectorAll('body[class*="pipeline-"]').forEach(b => {
      for (let i = b.classList.length - 1; i >= 0; i--) {
        const className = b.classList[i]
        if (className.startsWith('pipeline-')) {
          b.classList.remove(className)
        }
      }
    })

    if (model_name == 'kd20' && pipeline == 'diffusers') {
      kubin.notify.error('You cannot use a 2.0 model with the diffusers pipeline! Native pipeline will be used')
      pipeline = 'native'
    } else if (model_name == 'kd22' && pipeline == 'native') {
      kubin.notify.error('You cannot use a 2.2 model with the native pipeline! Diffusers pipeline will be used')
      pipeline = 'diffusers'
    }

    document.body.classList.add(`pipeline-${pipeline}-${model_name}`)
  }

  kubin.UI.resizablePanels = panelResize => {
    panelResize && Array.from(document.getElementsByClassName('block-params')).forEach(paramsBlock => {
      const anchor = document.createElement('div')
      const randomClass = `block-resizable-anchor-${kubin.randomString()}`
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
      anchor.remove()
    })
  }

  kubin.UI.verticalAlignment = verticalAlignment => {
    Array.from(document.getElementsByClassName('block-params')).forEach(paramsBlock => {
      verticalAlignment && paramsBlock.parentElement.classList.add('block-params-vertical-alignment')
      !verticalAlignment && paramsBlock.parentElement.classList.remove('block-params-vertical-alignment')
    })
  }

  kubin.UI.fullScreenUI = fullScreenUI => {
    fullScreenUI && document.body.classList.add('gradio-full')
    !fullScreenUI && document.body.classList.remove('gradio-full')
  }

  const imageIndices = {}
  kubin.UI.setImageIndex = (source) => {
    let position = -1
    Array.from(document.querySelectorAll(`.${source} .thumbnails .thumbnail-item`)).forEach((image, index) => {
      if (image.classList.contains('selected')) {
        position = index
      }
    })

    imageIndices[source] = position
    return position
  }

  kubin.UI.getImageIndex = (o, i, source) => {
    return [o, parseInt(imageIndices[source])]
  }

  kubin.UI.wakeAll = () => {
    document.querySelectorAll('[disabled]').forEach(disabled => {
      disabled.removeAttribute("disabled")
    })

    kubin.notify.warning('UI was waked up, disabled elements should be active now!')
  }

  kubin.UI.reveal = () => {
    document.getElementsByTagName('html')[0].classList.add('is-ready')
  }
})(window)
;
(global => {
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

  kubin.utils.reloadUI = () => {
    kubin.notify.success("App restart initiated, wait for page to reload")

    const url = window.location.origin
    const interval = 5

    const checkWithInterval = (url, interval) => {
      const check = () => {
        fetch(url).then(response => {
          response.ok && window.location.reload()
          !response.ok && setTimeout(check, interval * 1000)
        }).catch(_ => setTimeout(check, interval * 1000))
      }

      setTimeout(check, interval * 1000)
    }

    checkWithInterval(url, interval)
  }
})(window)
;
(() => {
  let justTriggered = false

  const closestDivisible = (number, denom) => {
    let q = Math.ceil(number / denom)
    return denom * q
  }

  const changeSize = (inputElement, size, adjustedSize) => {
    inputElement.value = size
    const changeEvent = new Event("input", {
      bubbles: true,
      cancelable: true
    })

    inputElement.dispatchEvent(changeEvent)
  }

  const addAdjustedSize = (element, size) => {
    let adjustedSize = element.nextSibling
    if (!adjustedSize) {
      adjustedSize = document.createElement('span')
      adjustedSize.className = 'ar-adjusted-size'
      adjustedSize.title = 'The size must be evenly divisible by 64 (as required by the MOVQ configuration), so the actual output size will be adjusted accordingly'
      element.parentNode.insertBefore(adjustedSize, element.nextSibling)
    }

    adjustedSize.textContent = `scaled: ${size}`
  }

  const removeAdjustedSize = element => {
    element.nextSibling && element.parentNode.removeChild(element.nextSibling)
  }

  kubin.UI.aspectRatio = {
    sizeChanged: (widthTarget, heightTarget, dimension, size, ar, denom) => {
      if (ar != "none") {
        if (justTriggered) {
          justTriggered = false
        } else {
          justTriggered = true

          const [w_ratio, h_ratio] = ar.trim().split(":").map(x => parseFloat(x))
          const ratio = dimension == 'width' ? h_ratio / w_ratio : w_ratio / h_ratio
          const newSize = parseInt(parseFloat(size) * ratio)
          const newAdjustedDependableSize = closestDivisible(newSize, denom)

          const inputElement = document.querySelector(`#${dimension == 'width' ? heightTarget : widthTarget} input[type=number]`)
          changeSize(inputElement, newSize, newAdjustedDependableSize)
          if (newAdjustedDependableSize != newSize) {
            addAdjustedSize(inputElement, newAdjustedDependableSize)
          } else {
            removeAdjustedSize(inputElement)
          }
        }
      }

      const newAdjustedSize = closestDivisible(size, denom)
      const inputSizeElement = document.querySelector(`#${dimension == 'width' ? widthTarget : heightTarget} input[type=number]`)
      if (newAdjustedSize != size) {
        addAdjustedSize(inputSizeElement, newAdjustedSize)
      } else {
        removeAdjustedSize(inputSizeElement)
      }
    }
  }
})()
;
(global => {
  kubin.mix = {
    createWidget: () => {

    }
  }
})(window)
;
(global => {
  kubin.visualizeAddedCondition = (labelSelector, triggerSelector, defaultValue) => {
    const label = document.querySelector(labelSelector)
    const trigger = document.querySelector(triggerSelector)

    if (label && trigger) {
      const added = document.createElement('span')

      added.textContent = 'enabled'
      added.className = 'added-condition'
      label.appendChild(added)

      let type = undefined

      trigger.tagName == 'INPUT' && trigger.type == 'checkbox' && (type = 'checkbox')
      trigger.tagName == 'FIELDSET' && (type = 'fieldset')

      trigger.addEventListener('change', e => {
        let value = undefined

        type == 'fieldset' && (value = e.target.value)
        type == 'checkbox' && (value = e.target.checked)
        added.classList[value === defaultValue ? 'remove' : 'add']('active')
      })
    }
  }

  kubin.UI.defaultConditions = () => {
    kubin.visualizeAddedCondition('.t2i_params .control-net .label-wrap span', '.t2i_params .control-net .cnet-enable [type=checkbox]', false)
    kubin.visualizeAddedCondition('.i2i_params .control-net .label-wrap span', '.i2i_params .control-net .cnet-enable [type=checkbox]', false)
    kubin.visualizeAddedCondition('.mix_params .control-net .label-wrap span', '.mix_params .control-net .cnet-enable [type=checkbox]', false)
    kubin.visualizeAddedCondition('.inpaint_params .control-net .label-wrap span', '.inpaint_params .control-net .cnet-enable [type=checkbox]', false)
    kubin.visualizeAddedCondition('.outpaint_params .control-net .label-wrap span', '.outpaint_params .control-net .cnet-enable [type=checkbox]', false)
  }
})(window)
;
(global => {
  kubin.requestApi = async (apiUrl, body) => {
    const origin = global.location.origin
    const url = `${origin}/${apiUrl}`
    const res = await fetch(url, {
      "headers": {
        "accept": "*/*",
        "content-type": "application/json"
      },
      "body": body || JSON.stringify({ data: [[], null], event_data: null, fn_index: 0, session_hash: kubin.client_id }),
      "method": "POST",
      "mode": "cors",
      "credentials": "include"
    })

    const json = await res.json()
    return json
  }
})(window)
;
(global => {
  let progressInterval = undefined

  let progressPanel = document.createElement("div")
  progressPanel.className = 'kubin-progress-panel'
  document.body.appendChild(progressPanel)

  kubin.UI.taskStarted = task => {
    let index = -1
    document.querySelectorAll(`.ui-tabs.left>.tab-nav button`).forEach((button, i) => button.textContent.includes(task) && (index = i + 1))
    index != -1 && (() => {
      const style = document.createElement("style")
      style.id = `working-indicator-${index}`
      style.textContent = `
        .ui-tabs.left>.tab-nav button:nth-child(${index})::after {
          content: " "; position: absolute; margin: auto; border: 10px solid #EAF0F6; border-radius: 50%; border-top: 10px solid var(--loader-color);
          width: 20px; height: 20px; animation: loader 2s linear infinite; margin-left: 7px; }
      `;

      document.head.appendChild(style)
      progressPanel.classList.add('active')
      progressInterval = setInterval(kubin.UI.progress, 2000)
    })()
  }

  kubin.UI.taskFinished = task => {
    let index = -1
    document.querySelectorAll(`.ui-tabs.left>.tab-nav button`).forEach((button, i) => button.textContent.includes(task) && (index = i + 1))
    index != -1 && (() => {
      const style = document.querySelector(`style#working-indicator-${index}`)
      style && style.parentNode.removeChild(style)
    })()

    progressInterval && clearInterval(progressInterval)
    progressPanel.classList.remove('active')
  }

  kubin.UI.progress = async () => {
    const progressInfo = await kubin.requestApi('api/progress')
    const taskInfo = progressInfo['data'][0]['progress']

    progressPanel.textContent = JSON.stringify(taskInfo)
  }

})(window)
;
