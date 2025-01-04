import { Element } from './element.js'

export class Label extends Element {
    init(name, config, children = []) {
        super.init(name, config, children, 'label')
        if (this.config.text) {
            this.dom.textContent = this.config.text; 
        }
    }

    static create(name, config, children = []) {
        const label = new Label()
        label.init(name, config, children)
        return label
    }
}