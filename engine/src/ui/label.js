import { Element } from './element.js'

export class Label extends Element {
    init(context, name, config, children = []) {
        super.init(context, name, config, children, 'label')
        if (this.config.text) {
            this.dom.textContent = this.config.text; 
        }
    }

    static create(context, name, config, children = []) {
        const label = new Label()
        label.init(context, name, config, children)
        return label
    }
}