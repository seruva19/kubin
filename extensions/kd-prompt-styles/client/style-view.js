((window) => {
    Array.from(document.querySelectorAll('.kd-prompt-styles-selector')).forEach(searchBox => {
        const searchInput = searchBox.querySelector('textarea')
        searchInput.addEventListener('input', event => {
            const value = event.target.value.toUpperCase()
            Array.from(searchBox.querySelectorAll(':scope .kd-styles-radiobutton-list label')).forEach(item => {
                const itemName = item.getAttribute('data-testid').toUpperCase()
                item.style.display = itemName.includes(value) ? 'initial' : 'none'
            })
        })
    })

    kubin.visualizeAddedCondition('.t2i_block .kd-prompt-styles .label-wrap span', '.t2i_block .kd-styles-radiobutton-list', 'none')
    kubin.visualizeAddedCondition('.i2i_block .kd-prompt-styles .label-wrap span', '.i2i_block .kd-styles-radiobutton-list', 'none')
    kubin.visualizeAddedCondition('.mix_block .kd-prompt-styles .label-wrap span', '.mix_block .kd-styles-radiobutton-list', 'none')
    kubin.visualizeAddedCondition('.inpaint_block .kd-prompt-styles .label-wrap span', '.inpaint_block .kd-styles-radiobutton-list', 'none')
    kubin.visualizeAddedCondition('.outpaint_block .kd-prompt-styles .label-wrap span', '.outpaint_block .kd-styles-radiobutton-list', 'none')
})(window)