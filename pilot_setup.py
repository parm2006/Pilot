"""Pilot's desktop setup wizard. Devices stay idle until calibration starts."""

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk


@dataclass(frozen=True)
class Preferences:
    primary_eye: str = "left"
    tracking: str = "head"
    voice: bool = True
    palm_mute: bool = True
    startup_listener: bool = False

    def __post_init__(self):
        if self.primary_eye not in ("left", "right") or self.tracking not in ("head", "eyes", "hybrid"):
            raise ValueError("Unknown Pilot control preference")
        if any(type(value) is not bool for value in (self.voice, self.palm_mute, self.startup_listener)):
            raise ValueError("Pilot switches must be true or false")

    def launch_flags(self):
        flags = ["--input", self.tracking]
        if self.primary_eye == "right":
            flags.append("--right")
        if not self.voice:
            flags.append("--no-voice")
        if not self.palm_mute:
            flags.append("--no-hand-mute")
        return flags


BG = "#101719"
PANEL = "#1A2426"
TEXT = "#F0F6F4"
MUTED = "#AFBFBC"
ACCENT = "#ADF4D4"
INK = "#142721"
LINE = "#435957"
FONT = "Segoe UI"
TRACKING_LABELS = {"head": "Head", "eyes": "Eyes", "hybrid": "Combined"}


class SetupWizard:
    def __init__(self, root, preferences, error):
        self.root = root
        self.result = None
        self.error = error
        self.step = 2 if error else 0
        self.eye = tk.StringVar(root, preferences.primary_eye)
        self.tracking = tk.StringVar(root, preferences.tracking)
        self.voice = tk.BooleanVar(root, preferences.voice)
        self.palm = tk.BooleanVar(root, preferences.palm_mute)
        self.startup = tk.BooleanVar(root, preferences.startup_listener)
        self.images = []

        root.title("Pilot Setup")
        root.configure(background=BG)
        root.resizable(False, False)
        root.bind("<Escape>", lambda _event: root.destroy())
        self.scale = root.winfo_fpixels("1i") / 96
        self.configure_style()
        self.content_width = self.pixels(440)
        self.shell = tk.Frame(root, bg=BG, padx=30, pady=22)
        self.shell.pack(fill="both", expand=True)
        self.render()
        root.update_idletasks()
        width = max(500, self.shell.winfo_reqwidth())
        height = self.shell.winfo_reqheight()
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (root.winfo_screenheight() - height) // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

    def pixels(self, size):
        return round(size * self.scale)

    def artwork(self, width, height, background, paint):
        # Draw curves above display resolution, then average down to screen pixels.
        size = (self.pixels(width), self.pixels(height))
        samples = 4
        bitmap = Image.new("RGB", tuple(length * samples for length in size), background)
        paint(ImageDraw.Draw(bitmap), self.scale * samples)
        photo = ImageTk.PhotoImage(bitmap.resize(size, Image.Resampling.BOX), master=self.root)
        self.images.append(photo)
        return photo

    def button_surface(self, fill, outline=None, line_width=1):
        def paint(draw, scale):
            draw.rounded_rectangle((0, 0, round(40 * scale) - 1, round(44 * scale) - 1),
                                   radius=8 * scale, fill=fill, outline=outline,
                                   width=round(line_width * scale))
        return self.artwork(40, 44, BG, paint)

    def switch_surface(self, selected, disabled=False):
        track = ACCENT if selected and not disabled else LINE
        thumb = INK if selected and not disabled else MUTED

        def paint(draw, scale):
            draw.rounded_rectangle((0, 0, round(38 * scale) - 1, round(22 * scale) - 1),
                                   radius=11 * scale, fill=track)
            left = 20 if selected else 4
            draw.ellipse((left * scale, 4 * scale, (left + 14) * scale, 18 * scale), fill=thumb)
        return self.artwork(38, 22, PANEL, paint)

    def brand_mark(self):
        def paint(draw, scale):
            draw.ellipse((1 * scale, 1 * scale, 33 * scale, 33 * scale), fill="#253D36")
            points = [(x * scale, y * scale) for x, y in ((10, 21), (22, 9), (19, 17), (24, 19), (11, 25))]
            draw.polygon(points, fill=ACCENT)
        return self.artwork(34, 34, BG, paint)

    def configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        def button(name, base, active, focus_ink, text):
            element = f"{name}.surface"
            is_primary = name == "Next"
            base_img = self.button_surface(base, None if is_primary else LINE)
            pressed = self.button_surface(active, None if is_primary else LINE)
            focussed = self.button_surface(base, focus_ink, line_width=2)
            selected_focus = self.button_surface(active, focus_ink, line_width=2)
            state = "selected" if name == "Choice" else "active"
            style.element_create(element, "image", base_img, (state, "focus", selected_focus), ("focus", focussed),
                                 (state, pressed), border=self.pixels(10), sticky="nsew")
            style.layout(f"{name}.TButton", [(element, {"sticky": "nsew", "children": [
                ("Button.padding", {"sticky": "nsew", "children": [
                    ("Button.label", {"sticky": "nsew"})]})]})])
            style.configure(f"{name}.TButton", foreground=text, anchor="center",
                            font=(FONT, 11, "bold"), padding=(12, 10))
        button("Next", ACCENT, "#C7FFE7", INK, INK)
        button("Back", PANEL, "#304843", ACCENT, TEXT)
        button("Choice", PANEL, "#304843", ACCENT, TEXT)
        style.map("Choice.TButton", foreground=[("selected", ACCENT)])
        style.element_create("Setting.switch", "image", self.switch_surface(False),
                             ("disabled", "selected", self.switch_surface(True, disabled=True)),
                             ("selected", self.switch_surface(True)), sticky="")
        style.layout("Setting.TCheckbutton", [
            ("Checkbutton.padding", {"sticky": "nsew", "children": [
                ("Setting.switch", {"side": "right", "sticky": ""}),
                ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                    ("Checkbutton.label", {"sticky": "w"})]}),
            ]}),
        ])
        style.configure("Setting.TCheckbutton", background=PANEL, foreground=TEXT,
                        font=(FONT, 11, "bold"), padding=(12, 4), focuscolor=ACCENT)
        style.map("Setting.TCheckbutton", background=[("active", PANEL)], foreground=[("disabled", MUTED)])
        self.mark = self.brand_mark()

    def label(self, parent, text, size=11, color=TEXT, weight="normal", bg=BG):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=(FONT, size, weight),
                        justify="left", anchor="w", wraplength=self.content_width - 16)

    def render(self):
        for child in self.shell.winfo_children():
            child.destroy()

        header = tk.Frame(self.shell, bg=BG)
        header.pack(fill="x")
        tk.Label(header, image=self.mark, bg=BG, bd=0).pack(side="left")
        self.label(header, "Pilot", 13, weight="bold").pack(side="left", padx=8)
        self.label(header, f"{self.step + 1} of 3", 10, MUTED).pack(side="right")

        progress = tk.Frame(self.shell, bg=BG)
        progress.pack(fill="x", pady=(13, 18))
        for index in range(3):
            segment = tk.Frame(progress, bg=ACCENT if index <= self.step else PANEL, height=3)
            segment.pack(side="left", fill="x", expand=True, padx=(0, 5 if index < 2 else 0))

        body = tk.Frame(self.shell, bg=BG, width=self.content_width, height=self.root.winfo_fpixels("3.9i"))
        body.pack(fill="x")
        body.pack_propagate(False)
        (self.controls, self.listening, self.calibration)[self.step](body)

        footer = tk.Frame(self.shell, bg=BG)
        footer.pack(fill="x", pady=(13, 0))
        if self.step:
            back = ttk.Button(footer, text="Back", style="Back.TButton", width=8, command=self.back)
            back.pack(side="left", padx=(0, 10))
            back.bind("<Return>", lambda _event: back.invoke())
        action = ttk.Button(footer, text="Begin calibration" if self.step == 2 else "Continue",
                            style="Next.TButton", command=self.advance)
        action.pack(side="left", fill="x", expand=True)
        action.bind("<Return>", lambda _event: action.invoke())
        self.root.after_idle(lambda: action.winfo_exists() and action.focus_set())

    def title(self, body, title, description):
        self.label(body, title, 24, weight="bold").pack(fill="x")
        self.label(body, description, color=MUTED).pack(fill="x", pady=(5, 17))

    def controls(self, body):
        self.title(body, "Make Pilot yours", "Set how you’ll move the pointer and click.")
        self.choices(body, "Primary click eye", self.eye, (("left", "Left"), ("right", "Right")))
        self.label(body, "Wink with this eye to left-click. The other eye right-clicks.",
                   10, MUTED).pack(fill="x", pady=(4, 16))
        self.choices(body, "Pointer tracking", self.tracking, TRACKING_LABELS.items())
        self.label(body, "Head tracking is a good place to start.", 10, MUTED).pack(fill="x", pady=(4, 0))

    def choices(self, body, title, variable, options):
        self.label(body, title, 11, weight="bold").pack(fill="x", pady=(0, 5))
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x")
        for index, (value, label) in enumerate(options):
            row.columnconfigure(index, weight=1, uniform=title)
            button = ttk.Radiobutton(row, text=label, value=value, variable=variable,
                                     style="Choice.TButton")
            button.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 5, 0))
            button.bind("<Return>", lambda _event, control=button: control.invoke())

    def listening(self, body):
        self.title(body, "Voice and startup", "Choose when Pilot can listen.")
        self.setting(body, "Voice commands", "Use the default microphone while Pilot is open.",
                     self.voice, command=self.update_palm)
        self.palm_toggle = self.setting(body, "Mute with an open palm",
                                        "Face your palm toward the camera and spread your fingers.", self.palm)
        self.update_palm()
        self.setting(body, "Start wake listener with Windows",
                     "Say “Pilot” to open the app after logging into Windows.", self.startup)

    def setting(self, body, title, note, variable, command=None):
        card = tk.Frame(body, bg=PANEL, padx=6, pady=4)
        card.pack(fill="x", pady=(0, 8))
        toggle = ttk.Checkbutton(card, text=title, variable=variable,
                                 style="Setting.TCheckbutton", command=command)
        toggle.pack(fill="x")
        toggle.bind("<Return>", lambda _event: toggle.invoke())
        self.label(card, note, 10, MUTED, bg=PANEL).pack(fill="x", padx=12, pady=(0, 4))
        return toggle

    def update_palm(self):
        self.palm_toggle.state(["!disabled"] if self.voice.get() else ["disabled"])

    def calibration(self, body):
        self.title(body, "Ready to calibrate", "Pilot opens your camera for two short exercises.")
        panel = tk.Frame(body, bg=PANEL, padx=15, pady=11)
        panel.pack(fill="x")
        self.label(panel, "1   Follow the dots", 11, weight="bold", bg=PANEL).pack(fill="x")
        self.label(panel, "Turn your head toward each dot and look at it.",
                   10, MUTED, bg=PANEL).pack(fill="x", pady=(3, 10))
        self.label(panel, "2   Practice each wink", 11, weight="bold", bg=PANEL).pack(fill="x")
        self.label(panel, "Keep both eyes open, then close one eye at a time.",
                   10, MUTED, bg=PANEL).pack(fill="x", pady=(3, 0))
        self.label(body, "Esc cancels calibration. After setup, F7 recenters and F9 recalibrates.",
                   10, MUTED).pack(fill="x", pady=(11, 0))
        if self.error:
            brief = self.error.splitlines()[0][:140]
            self.label(body, brief, 10, "#FFB6A6").pack(fill="x", pady=(6, 0))
        else:
            summary = f"{self.eye.get().title()} wink  ·  {TRACKING_LABELS[self.tracking.get()]} tracking"
            summary += "  ·  Voice on" if self.voice.get() else "  ·  Voice off"
            self.label(body, summary, 10, ACCENT).pack(fill="x", pady=(15, 0))

    def back(self):
        self.step -= 1
        self.render()

    def advance(self):
        if self.step < 2:
            self.step += 1
            self.render()
            return
        self.result = Preferences(self.eye.get(), self.tracking.get(), self.voice.get(),
                                  self.palm.get(), self.startup.get())
        self.root.destroy()


def show_setup(preferences, error=""):
    root = tk.Tk()
    wizard = SetupWizard(root, preferences, error)
    root.mainloop()
    return wizard.result
