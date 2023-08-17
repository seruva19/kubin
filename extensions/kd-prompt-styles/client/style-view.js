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
})(window)