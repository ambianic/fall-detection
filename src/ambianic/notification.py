"""Utilities to send notifications."""

import logging
import apprise
import os
import ambianic

log = logging.getLogger(__name__)

UI_BASEURL = "https://ui.ambianic.ai"

class Notification:
    def __init__(self, event:str = "detection", data:dict = {}, providers:list = ["all"]):
        self.event: str = event
        self.providers: list = providers
        self.title: str = None
        self.message: str = None
        self.attach: list = []
        self.data: dict = data

    def add_attachments(self, *args):
        self.attach.append(*args)

    def to_dict(self) -> dict:
        return dict(vars(self))


class NotificationHandler:
    def __init__(self, config: dict = None):
        if config is None:
            config = ambianic.config
        self.apobj = apprise.Apprise(debug=True)
        self.config = config.get("notifications", {})
        for name, cfg in self.config.items():
            providers = cfg.get("providers", [])
            for provider in providers:
                if not self.apobj.add(provider, tag=name):
                    log.warning(
                        "Failed to add notification provider: %s=%s" 
                            % (name, provider)
                    )

    def send(self, notification: Notification):

        templates = self.config.get("templates", {})

        title = notification.title
        if title is None:
            title = templates.get("title", "[Ambianic.ai] New ${event} event" )

        message = notification.message
        if message is None:
            message = templates.get(
                "message", "New ${event} recognized"
            )

        attachments = []
        for a in notification.attach:
            if not os.path.exists(a) or not os.path.isfile(a):
                log.warning("Attachment is not a valid file %s")
                continue
            attachments.append(a)

        template_args = {
            "event_type": notification.event,
            "event": notification.data.get("label", notification.event),
            "event_details_url": "%s/%s" % (UI_BASEURL, notification.data.get("id", ""))
        }
        template_args = {**template_args, **notification.data}

        for key, value in template_args.items():
            k = "${%s}" % (str(key))
            v = str(value)
            title = title.replace(k, v)
            message = message.replace(k, v)

        for provider in notification.providers:
            cfg = self.config.get(provider, None)
            if cfg is None:
                log.warning("Skip unknown provider %s" % provider)
                continue
            
            include_attachments = cfg.get("include_attachments", False)
            ok = self.apobj.notify(
                message, 
                title=title,
                tag=provider, 
                attach=attachments if include_attachments else [],
            )
            if ok:
                log.debug(
                    "Sent notification for %s to %s" % 
                    (notification.event, provider)
                )
            else:
                log.warning("Error sending notification for %s to %s" %
                          (notification.event, provider))
