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
