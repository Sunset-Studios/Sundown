import { Element } from './element.js'

export class Button extends Element {
    init(context, name, config, children = []) {
        super.init(context, name, config, children, 'button')

        if (this.config.icon) {
            const url = new URL(`${this.config.icon}`, window.location.href);
            const icon = document.createElement('img')
            icon.src = url.href
            icon.style.width = this.config.style.width
            icon.style.height = this.config.style.height
            this.dom.appendChild(icon)
        } else {
            this.dom.textContent = this.config.text; 
        }
    }

    static create(context, name, config, children = []) {
        const button = new Button()
        button.init(context, name, config, children)
        return button
    }
}