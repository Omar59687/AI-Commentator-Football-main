from core.events import Event


class EventManager:

    def __init__(self):

        self.events = []

    def add_event(self, event: Event):

        self.events.append(event)

    def get_events(self):

        return sorted(
            self.events,
            key=lambda e: e.timestamp
        )