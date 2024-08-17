import { Element } from './element.js'

export class Button extends Element {
    icon = null;

    init(context, name, config, children = []) {
        super.init(context, name, config, children, 'button')

        if (this.config.icon) {
            const url = new URL(`${this.config.icon}`, window.location.href);
            this.icon = document.createElement('img')
            this.icon.src = url.href
            Object.assign(this.icon.style, this.config.style)
            this.dom.appendChild(this.icon)
        } else {
            this.dom.textContent = this.config.text; 
        }
    }

    apply_style(style, reset = false) {
        super.apply_style(style, reset);
        if (this.icon) {
            if (reset) {
                this.icon.style = {};
            }
            Object.assign(this.icon.style, style);
        }
    }

    static create(context, name, config, children = []) {
        const button = new Button()
        button.init(context, name, config, children)
        return button
    }
}