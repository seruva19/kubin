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
