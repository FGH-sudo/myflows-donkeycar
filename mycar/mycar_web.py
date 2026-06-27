# -*- coding: utf-8 -*-
"""
Serve the Donkey web UI with mycar 自带的 static（例如定制 main.js 的 WASD/油门限制）。
回退到 donkey 默认 static：若 mycar/web_static 不存在或缺少 main.js。
"""
import os

from socket import gethostname
from tornado.web import Application, RedirectHandler, StaticFileHandler

from donkeycar.parts.web_controller import web


class MycarLocalWebController(web.LocalWebController):
    def __init__(self, port=8887, mode='user'):
        web.logger.info('Starting Donkey Server (mycar web_static override)...')
        mycar_dir = os.path.dirname(os.path.abspath(__file__))
        my_static = os.path.join(mycar_dir, 'web_static')
        if not os.path.isfile(os.path.join(my_static, 'main.js')):
            pkg = os.path.dirname(os.path.abspath(web.__file__))
            my_static = os.path.join(pkg, 'templates', 'static')
        self.static_file_path = my_static
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.mode_latch = None
        self.recording = False
        self.recording_latch = None
        self.buttons = {}
        self.port = port
        self.num_records = 0
        self.wsclients = []
        self.loop = None

        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", web.DriveAPI),
            (r"/wsDrive", web.WebSocketDriveAPI),
            (r"/wsCalibrate", web.WebSocketCalibrateAPI),
            (r"/calibrate", web.CalibrateHandler),
            (r"/video", web.VideoAPI),
            (r"/wsTest", web.WsTest),
            (r"/static/(.*)", StaticFileHandler, {"path": self.static_file_path}),
        ]
        settings = {'debug': True}
        Application.__init__(self, handlers, **settings)
        web.logger.info(
            f"You can now go to {gethostname()}.local:{port} to "
            f"drive your car (static={self.static_file_path})."
        )
