"""Aspecto «Windows 2000» con un toque futurista (ventanas grises con barra azul y relieve, barra de
tareas abajo; la consola de las IA, oscura y con brillo cian por dentro).

Es solo aspecto (CSS sobre los elementos de Streamlit y algo de HTML propio): no cambia ningún cálculo.
Inspirado en el estilo de la época, sin logotipos ni marcas. Las ventanas no se pueden arrastrar
(Streamlit no lo permite) y en el móvil la letra puede variar (Tahoma no existe en todos los equipos).
"""
import html

import streamlit as st

# Botones de la barra de título (minimizar, maximizar, cerrar): adorno dibujado en SVG.
_TITLE_BUTTONS = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='54' height='14'>"
    + "".join("<g transform='translate(%d,0)'><rect x='0' y='0' width='16' height='14' fill='%%23d4d0c8'/>"
              "<path d='M0 13.5H15.5V0' stroke='%%23404040' fill='none'/><path d='M0.5 13V0.5H15' stroke='white' fill='none'/>"
              "%s</g>" % (x, glyph) for x, glyph in (
                  (0, "<rect x='4' y='9' width='6' height='2' fill='black'/>"),
                  (18, "<rect x='3.5' y='2.5' width='8' height='7' fill='none' stroke='black'/><rect x='3' y='2' width='9' height='2' fill='black'/>"),
                  (38, "<path d='M4 3L11 10M11 3L4 10' stroke='black' stroke-width='1.6'/>")))
    + "</svg>")

CSS = """<style>
:root{--w-face:#d4d0c8;--w-hi:#ffffff;--w-lt:#e9e7e2;--w-sh:#808080;--w-dk:#404040;--w-t1:#0a246a;--w-t2:#a6caf0;
 --hud:#050b1a;--hud-line:#1d4a63;--cy:#5ff3ff;--cy2:#1aa3c9;--ok:#39ff88;--bad:#ff5f6d;--warn:#ffcf4a;
 --w-font:Tahoma,Verdana,"Segoe UI","DejaVu Sans",sans-serif;--hud-font:Consolas,"Lucida Console","DejaVu Sans Mono",monospace}
html{font-size:15px}
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select{font-family:var(--w-font)}
/* ---------- escritorio: azul profundo con cuadrícula tenue (el toque futurista) ---------- */
.stApp{background:
  linear-gradient(rgba(95,243,255,.05) 1px,transparent 1px) 0 0/32px 32px,
  linear-gradient(90deg,rgba(95,243,255,.05) 1px,transparent 1px) 0 0/32px 32px,
  radial-gradient(ellipse at 70% 15%,#16407a 0,#0b2350 45%,#050e24 100%) fixed !important;color:#000}
[data-testid="stHeader"]{background:transparent !important}
/* ---------- ventana principal ---------- */
[data-testid="stMainBlockContainer"]{background:var(--w-face);border:1px solid;
  border-color:var(--w-hi) var(--w-dk) var(--w-dk) var(--w-hi);
  box-shadow:inset -1px -1px var(--w-sh),inset 1px 1px var(--w-lt),6px 8px 26px rgba(0,0,0,.55);
  margin:3rem auto 3.4rem;padding:0 1.1rem 1.6rem !important;width:calc(100% - 2rem)}
[data-testid="stMainBlockContainer"]::before{content:"🌌  CMS-80 Cosmic Materials Scout";display:block;
  margin:3px calc(-1.1rem + 3px) 10px;padding:4px 70px 4px 8px;color:#fff;font:bold 14px var(--w-font);
  background:url("__BUTTONS__") no-repeat right 4px center,linear-gradient(90deg,var(--w-t1),var(--w-t2))}
h2#cms-80-cosmic-materials-scout{font-size:0 !important;height:0;margin:0 !important;padding:0 !important;overflow:hidden}
/* títulos de sección: barra de título azul */
[data-testid="stMainBlockContainer"] h3, [data-testid="stSidebar"] h3{
  background:linear-gradient(90deg,var(--w-t1),var(--w-t2));color:#fff !important;font:bold 14px var(--w-font) !important;
  padding:4px 8px !important;margin:14px 0 8px !important;letter-spacing:.2px}
[data-testid="stMainBlockContainer"] h3 a, [data-testid="stMainBlockContainer"] h3 svg{color:#fff !important}
[data-testid="stMainBlockContainer"] h4{color:var(--w-t1)}
/* ventanas interiores (contenedores con clave win_…) */
[class*="st-key-win_"]{background:var(--w-face);border:1px solid !important;border-radius:0 !important;
  border-color:var(--w-hi) var(--w-dk) var(--w-dk) var(--w-hi) !important;
  box-shadow:inset -1px -1px var(--w-sh),inset 1px 1px var(--w-lt),3px 4px 0 rgba(0,0,0,.18);padding:3px 10px 10px !important}
[class*="st-key-win_"] h3{margin-top:0 !important;margin-left:-7px !important;margin-right:-7px !important}
/* ---------- botones en relieve ---------- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, .stLinkButton a{
  background:var(--w-face) !important;color:#000 !important;border-radius:0 !important;border:1px solid !important;
  border-color:var(--w-hi) var(--w-dk) var(--w-dk) var(--w-hi) !important;
  box-shadow:inset -1px -1px var(--w-sh),inset 1px 1px var(--w-lt) !important;font-family:var(--w-font) !important}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover{color:var(--w-t1) !important}
.stButton button:active, .stDownloadButton button:active, .stFormSubmitButton button:active{
  border-color:var(--w-dk) var(--w-hi) var(--w-hi) var(--w-dk) !important;box-shadow:inset 1px 1px var(--w-sh) !important}
[data-testid^="stBaseButton-primary"]{font-weight:bold !important;outline:1px solid #000 !important;outline-offset:0}
.stApp button:disabled{color:var(--w-sh) !important;text-shadow:1px 1px var(--w-hi)}
.stApp button:focus-visible{outline:1px dotted #000 !important;outline-offset:-4px}
/* ---------- campos hundidos ---------- */
[data-baseweb="input"], [data-baseweb="select"]>div, [data-baseweb="textarea"], [data-testid="stNumberInputContainer"]{
  background:#fff !important;border-radius:0 !important;border:1px solid !important;
  border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh) !important;box-shadow:inset 1px 1px var(--w-dk) !important}
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea{background:#fff !important;color:#000 !important}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stImage"] img, [data-testid="stCode"]{
  border:1px solid;border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);box-shadow:inset 1px 1px var(--w-dk)}
/* ---------- grupos, pestañas, avisos ---------- */
[data-testid="stExpander"] details{border-radius:0 !important;border:1px solid var(--w-sh) !important;
  box-shadow:1px 1px var(--w-hi),inset 1px 1px var(--w-hi);background:var(--w-face)}
[data-testid="stExpander"] summary{font-weight:bold}
[data-testid="stExpander"] summary:hover{color:var(--w-t1)}
[data-baseweb="tab-list"]{gap:2px;border-bottom:1px solid var(--w-hi)}
[data-baseweb="tab-list"] button[role="tab"]{background:var(--w-face);border:1px solid;border-bottom:none;
  border-color:var(--w-hi) var(--w-dk) transparent var(--w-hi);border-radius:3px 3px 0 0;padding:4px 10px;margin-top:3px}
[data-baseweb="tab-list"] button[aria-selected="true"]{margin-top:0;padding-bottom:7px;font-weight:bold}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none}
[data-testid="stTabs"] [role="tablist"]{gap:2px;border-bottom:1px solid var(--w-hi)}
[data-testid="stTab"]{background:var(--w-face);border:1px solid;border-bottom:none;border-radius:3px 3px 0 0;
  border-color:var(--w-hi) var(--w-dk) transparent var(--w-hi);box-shadow:inset -1px 0 var(--w-sh);padding:4px 10px;margin-top:3px}
[data-testid="stTab"][aria-selected="true"]{margin-top:0;padding-bottom:7px}
[data-testid="stTab"][aria-selected="true"] p{font-weight:bold}
[data-testid="stTab"]>div[data-rac]{display:none}
[data-testid="stTabPanel"]{border:1px solid;border-top:none;border-color:transparent var(--w-dk) var(--w-dk) var(--w-hi);
  box-shadow:inset -1px -1px var(--w-sh);padding:10px}
[data-testid="stAlert"]>div{border-radius:0 !important;border:1px solid var(--w-dk)}
[data-testid="stChatMessage"]{background:#fff;border:1px solid;border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);
  border-radius:0}
[data-testid="stMetric"]{background:#fff;border:1px solid;border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);padding:6px 10px}
[data-testid="stMetricValue"]{color:var(--w-t1) !important}
.cms-prov{border-left:3px solid var(--w-t1);background:#fff;padding:.4rem .8rem;margin:.4rem 0;font-size:.92rem}
/* ---------- barra lateral: banda azul vertical, como un menú de inicio ---------- */
[data-testid="stSidebar"]{background:var(--w-face) !important;border-right:1px solid var(--w-dk);
  box-shadow:inset -1px 0 var(--w-sh),inset 1px 0 var(--w-hi)}
[data-testid="stSidebarContent"]{position:relative;padding-left:24px}
[data-testid="stSidebarContent"]::before{content:"CMS-80  Scout";position:absolute;left:0;top:0;bottom:0;width:24px;
  background:linear-gradient(0deg,var(--w-t1),#3a6ea5 60%,var(--w-t2));color:#fff;font:bold 15px var(--w-font);
  writing-mode:vertical-rl;transform:rotate(180deg);padding:14px 0;letter-spacing:1px}
/* ---------- barra de tareas ---------- */
.w2k-taskbar{position:fixed;left:0;right:0;bottom:0;height:32px;z-index:1000100;background:var(--w-face);
  border-top:1px solid var(--w-hi);box-shadow:0 -1px var(--w-lt);display:flex;align-items:center;gap:4px;padding:0 3px;
  font:13px var(--w-font);color:#000}
.w2k-taskbar a{color:#000 !important;text-decoration:none !important}
.w2k-start, .w2k-task{display:inline-block;padding:3px 9px;border:1px solid;border-color:var(--w-hi) var(--w-dk) var(--w-dk) var(--w-hi);
  box-shadow:inset -1px -1px var(--w-sh),inset 1px 1px var(--w-lt);white-space:nowrap;background:var(--w-face)}
.w2k-start{font-weight:bold}
.w2k-task{max-width:170px;overflow:hidden;text-overflow:ellipsis}
.w2k-tray{margin-left:auto;margin-right:150px;padding:3px 8px;border:1px solid;
  border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);display:flex;gap:10px;white-space:nowrap;overflow:hidden}
@media (max-width:700px){.w2k-task{display:none}.w2k-tray{margin-right:4px}
  [data-testid="stMainBlockContainer"]{width:calc(100% - .6rem)}}
/* chips de IA (semáforo) sobre gris */
.ai-chip{display:inline-block;padding:1px 8px;margin:2px 4px 2px 0;background:#fff;color:#000;border:1px solid;border-radius:0;
  border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);font-size:.85rem;white-space:nowrap}
.ai-chip.on{font-weight:bold;color:#05501f}.ai-chip.err{color:#9a0000}.ai-chip.wait{color:#6b4e00}.ai-chip.off{color:#6b6b6b}
/* ---------- ventana del consejo de IA: Windows 2000 por fuera, consola futurista por dentro ---------- */
.w2k-title{display:flex;align-items:center;gap:6px;margin:0 -7px 3px;padding:3px 4px 3px 8px;color:#fff;font:bold 14px var(--w-font);
  background:linear-gradient(90deg,var(--w-t1),var(--w-t2))}
.w2k-title .t{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.w2k-title .b{width:54px;height:14px;background:url("__BUTTONS__") no-repeat;flex:none}
.st-key-ai_console{background:var(--hud) !important;color:#cfefff;border:1px solid;border-radius:0;
  border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);padding:12px !important;position:relative;
  box-shadow:inset 0 0 28px rgba(95,243,255,.12),inset 1px 1px var(--w-dk);font-family:var(--hud-font)}
.st-key-ai_console::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,rgba(255,255,255,.03) 0 1px,transparent 1px 3px)}
.st-key-ai_console p, .st-key-ai_console li, .st-key-ai_console label, .st-key-ai_console span,
.st-key-ai_console [data-testid="stCaptionContainer"], .st-key-ai_console [data-testid="stCaptionContainer"] p{color:#cfefff}
.st-key-ai_console strong{color:#fff}
.st-key-ai_console .stButton button, .st-key-ai_console .stButton button *{color:#000 !important}
.st-key-ai_console a{color:var(--cy) !important}
.st-key-ai_console [data-testid="stCaptionContainer"]{opacity:.85}
.st-key-ai_console [class*="st-key-aicard"]{background:rgba(10,30,60,.6) !important;border:1px solid var(--hud-line) !important;
  border-radius:0 !important;padding:8px 10px !important}
.st-key-ai_console [data-testid="stExpander"] details{background:transparent;border-color:var(--hud-line) !important;box-shadow:none}
.st-key-ai_console [data-testid="stExpander"] summary, .st-key-ai_console [data-testid="stExpander"] summary *{color:var(--cy) !important}
.st-key-ai_console [data-testid="stAlert"]>div{background:rgba(10,30,60,.6);color:#cfefff;border-color:var(--hud-line)}
.hud-mods{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.hud-mod{flex:1 1 140px;border:1px solid var(--hud-line);background:linear-gradient(180deg,#0a1a33,#071226);padding:5px 8px}
.hud-mod b{color:#fff}.hud-mod small{display:block;color:#8fb9cc;font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.led{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle;background:#5a6b7a}
.led.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}.led.err{background:var(--bad);box-shadow:0 0 8px var(--bad)}
.led.wait{background:var(--warn);box-shadow:0 0 6px var(--warn)}.led.zz{background:#7a6cff;box-shadow:0 0 6px #7a6cff}
.hud-line{color:var(--cy);margin:2px 0 6px}
.hud-prog{height:16px;border:1px solid;border-color:var(--w-sh) var(--w-hi) var(--w-hi) var(--w-sh);background:#020713;padding:2px;margin:4px 0 8px}
.hud-prog>span{display:block;height:100%;width:var(--fill,100%);
  background:repeating-linear-gradient(90deg,var(--cy) 0 9px,transparent 9px 11px);box-shadow:0 0 6px rgba(95,243,255,.45)}
.hud-prog.run>span{animation:hudfill 2.4s steps(24) infinite}
@keyframes hudfill{from{width:0}to{width:100%}}
.hud-head{color:var(--cy);font-weight:bold;letter-spacing:.4px}
.hud-head .m{color:#8fb9cc;font-weight:normal;letter-spacing:0}
.ai-ok{color:var(--ok)}.ai-warn{color:var(--warn)}.ai-head{font-size:.85rem;color:#8fb9cc}
.cite{display:inline-block;padding:0 4px;border:1px solid;font-size:.82rem;margin:0 1px;line-height:1.25}
.cite.ok{color:var(--ok) !important;border-color:#1f7a4a}.cite.bad{color:var(--bad) !important;border-color:#8a2a35;text-decoration:line-through}
.hud-checks{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.8rem;margin-top:4px}
.hud-checks .y{color:var(--ok)}.hud-checks .n{color:var(--bad)}
.hud-cons{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.hud-cons .bar{flex:1 1 120px;height:8px;background:#0a1a33;border:1px solid var(--hud-line)}
.hud-cons .bar i{display:block;height:100%;background:var(--ok);box-shadow:0 0 6px var(--ok)}
</style>""".replace("__BUTTONS__", _TITLE_BUTTONS)


def apply():
    """El tema completo. Se llama una vez, arriba del todo."""
    st.markdown(CSS, unsafe_allow_html=True)


def window_title(text, icon="", note=""):
    """Barra de título azul de una ventana (texto escapado: puede llevar la pregunta de la persona)."""
    st.markdown('<div class="w2k-title"><span>%s</span><span class="t">%s%s</span><span class="b"></span></div>'
                % (html.escape(icon), html.escape(text), (" — " + html.escape(note)) if note else ""),
                unsafe_allow_html=True)


def tasks_for(field_loaded):
    """Accesos de la barra de tareas: solo a secciones que están en pantalla (sin archivo cargado, la
    app termina en «1. Fuente de datos»)."""
    out = [("🧠 Guía", "#guia"), ("📂 1. Fuente", "#fuente")]
    if field_loaded:
        out += [("🖼️ 2. Región", "#region"), ("🔬 4. Analizar y resultados", "#analizar")]
    return out


def taskbar(tray, tasks):
    """Barra de tareas fija abajo: «Inicio» (sube a la guía), accesos a las secciones y, en la bandeja,
    el semáforo de cada IA. `tray`: lista de (luz, nombre); `tasks`: lista de (texto, #ancla)."""
    items = "".join('<span>%s %s</span>' % (html.escape(light), html.escape(name)) for light, name in tray) \
        or "<span>IA: sin conectar</span>"
    links = "".join('<a class="w2k-task" href="%s">%s</a>' % (html.escape(href), html.escape(label)) for label, href in tasks)
    st.markdown('<div class="w2k-taskbar"><a class="w2k-start" href="#guia">🌌 Inicio</a>%s'
                '<div class="w2k-tray" title="Semáforo de las IA">%s</div></div>' % (links, items),
                unsafe_allow_html=True)
