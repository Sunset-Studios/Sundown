export default class ExecutionQueue {
    constructor() {
        this.executions = [];
        this.execution_delays = [];
    }

    push_execution(execution, execution_frame_delay = 0) {
        this.executions.push(execution);
        this.execution_delays.push(execution_frame_delay);
    }

    remove_execution(execution) {
        const index = this.executions.indexOf(execution);
        if (index !== -1) {
            this.executions.splice(index, 1);
            this.execution_delays.splice(index, 1);
        }
    }

    update() {
        for (let i = this.executions.length - 1; i >= 0; --i) {
            if (--this.execution_delays[i] < 0) {
                this.executions[i]();
                this.executions.splice(i, 1);
                this.execution_delays.splice(i, 1);
            }
        }
    }

    flush() {
        for (let i = this.executions.length - 1; i >= 0; --i) {
            this.executions[i]();
        }
        this.executions = [];
        this.execution_delays = [];
    }
}
