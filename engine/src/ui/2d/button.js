import { Element } from './element.js'

export class Button extends Element {
    icon = null;

    init(name, config, children = []) {
        super.init(name, config, children, 'button')
        
        this.config.allows_cursor_events = true;

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

    static create(name, config, children = []) {
        const button = new Button()
        button.init(name, config, children)
        return button
    }
}