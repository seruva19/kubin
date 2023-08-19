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
