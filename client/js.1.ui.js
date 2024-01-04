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
