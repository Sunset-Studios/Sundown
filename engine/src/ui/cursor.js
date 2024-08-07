import { Element } from './element.js'
import { InputProvider } from '../input/input_provider.js'
import { InputRange } from '../input/input_types.js'

export class Cursor extends Element {
    context = null

    init(context, name, config, children = []) {
        super.init(context, name, config, children, 'cursor')

        this.context = context

        if (this.config.icon) {
            const url = new URL(`${this.config.icon}`, window.location.href);
            const icon = document.createElement('img')
            icon.src = url.href
            icon.style.width = this.config.style.width
            icon.style.height = this.config.style.height
            this.dom.appendChild(icon)
        }
    }

    update(delta_time) {
        super.update(delta_time)

        const x = InputProvider.get().get_range(InputRange.M_xabs);
        const y = InputProvider.get().get_range(InputRange.M_yabs);
    
        this.dom.style.left = x + 'px'
        this.dom.style.top = y + 'px'
    }

    static create(context, name, config, children = []) {
        const cursor = new Cursor()
        cursor.init(context, name, config, children)
        return cursor
    }
}