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
