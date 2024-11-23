class DispatcherEvent {
  constructor(event_name = "") {
    this.event_name = event_name;
    this.callbacks = [];
  }

  register(callback) {
    this.callbacks.push(callback);
  }

  unregister(callback) {
    const index = this.callbacks.indexOf(callback);
    if (index != -1) {
      this.callbacks.splice(index, 1);
    }
  }

  broadcast(...data) {
    const callbacks = this.callbacks.slice(0);
    for (let i = 0; i < callbacks.length; ++i) {
      callbacks[i](...data);
    }
  }
}

export class Dispatcher {
  constructor() {
    this.events = {};
  }

  dispatch(event_name, ...data) {
    if (this.events[event_name]) {
      this.events[event_name].broadcast(...data);
    }
  }

  on(event_name, callback) {
    if (!(event_name in this.events)) {
      this.events[event_name] = new DispatcherEvent(event_name);
    }
    this.events[event_name].register(callback);
  }

  off(event_name, callback) {
    if (this.events[event_name]) {
      this.events[event_name].unregister(callback);
    }
    if (this.events[event_name].callbacks.length === 0) {
      delete this.events[event_name];
    }
  }
}

export const global_dispatcher = new Dispatcher();
