"""
Inline SVG mascot for the PVT Studio header.

A stylized "PVT scientist" oil drop with a measuring beaker — small, lightweight,
and renders inline without external image dependencies.
"""

MASCOT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100" width="180" height="90">
  <!-- Background panel -->
  <rect x="0" y="0" width="200" height="100" fill="#00243D" rx="6"/>

  <!-- Beaker on the left -->
  <g transform="translate(20, 20)">
    <!-- Beaker outline -->
    <path d="M 8 5 L 8 30 Q 8 50 28 50 Q 48 50 48 30 L 48 5"
          stroke="#FFE7D6" stroke-width="2" fill="none"/>
    <!-- Beaker rim -->
    <line x1="4" y1="5" x2="52" y2="5" stroke="#FFE7D6" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Liquid inside -->
    <path d="M 9 25 Q 9 48 28 48 Q 47 48 47 25 L 47 25 Z"
          fill="#EB0037" opacity="0.85"/>
    <!-- Bubble in liquid -->
    <circle cx="22" cy="38" r="2.5" fill="#FFE7D6" opacity="0.7"/>
    <circle cx="34" cy="34" r="1.8" fill="#FFE7D6" opacity="0.7"/>
    <circle cx="28" cy="42" r="1.5" fill="#FFE7D6" opacity="0.6"/>
    <!-- Bubbles rising above beaker -->
    <circle cx="28" cy="1" r="1.5" fill="#FFE7D6" opacity="0.7"/>
    <circle cx="32" cy="-3" r="1.0" fill="#FFE7D6" opacity="0.5"/>
  </g>

  <!-- Oil-drop mascot on the right -->
  <g transform="translate(120, 18)">
    <!-- Drop body -->
    <path d="M 30 4 Q 8 24 8 48 Q 8 70 30 70 Q 52 70 52 48 Q 52 24 30 4 Z"
          fill="#EB0037" stroke="#FFE7D6" stroke-width="1.5"/>
    <!-- Eyes -->
    <circle cx="22" cy="40" r="3.5" fill="#FFFFFF"/>
    <circle cx="38" cy="40" r="3.5" fill="#FFFFFF"/>
    <circle cx="22" cy="40" r="1.8" fill="#00243D"/>
    <circle cx="38" cy="40" r="1.8" fill="#00243D"/>
    <!-- Eye shine -->
    <circle cx="23" cy="39" r="0.6" fill="#FFFFFF"/>
    <circle cx="39" cy="39" r="0.6" fill="#FFFFFF"/>
    <!-- Smiling mouth -->
    <path d="M 22 52 Q 30 58 38 52" stroke="#FFFFFF" stroke-width="1.8"
          fill="none" stroke-linecap="round"/>
    <!-- Tiny lab goggles strap (the eyebrows) -->
    <path d="M 17 35 L 28 32 L 32 32 L 43 35" stroke="#9DBA00"
          stroke-width="1.6" fill="none" stroke-linecap="round"/>
  </g>

  <!-- PVT label -->
  <text x="100" y="92" text-anchor="middle" fill="#FFE7D6"
        font-family="Helvetica, Arial, sans-serif" font-size="10"
        font-weight="bold" letter-spacing="2px">P V T   S T U D I O</text>
</svg>
"""


def header_with_mascot(title, subtitle):
    """Banner with SVG mascot on the right."""
    return f"""
<div style="background: linear-gradient(90deg, #00243D 0%, #1B3A5B 100%);
            padding: 1rem 1.5rem; border-radius: 4px; margin-bottom: 1rem;
            border-left: 6px solid #EB0037; display: flex; align-items: center;
            justify-content: space-between; gap: 1rem;">
  <div style="flex: 1;">
    <h1 style="color: #FFFFFF; margin: 0; font-size: 1.6rem; font-weight: 600;
               letter-spacing: -0.01em;">{title}</h1>
    <p style="color: #FFE7D6; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
      {subtitle}
    </p>
  </div>
  <div style="flex-shrink: 0;">
    {MASCOT_SVG}
  </div>
</div>
"""
