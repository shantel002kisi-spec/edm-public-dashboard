from pathlib import Path

path = Path("app.py")
text = path.read_text(encoding="utf-8")

start_marker = '    poster_high, poster_low = st.columns(2)\n'
end_marker = (
    '    st.caption(\n'
    '        "Poster-aligned dissertation results. The company ranking uses the audited "\n'
    '        "counted-spill totals already built into this dashboard."\n'
    '    )\n'
)

start = text.find(start_marker)
if start == -1:
    raise SystemExit("Could not find poster card section start")

end_start = text.find(end_marker, start)
if end_start == -1:
    raise SystemExit("Could not find poster card section end")
end = end_start + len(end_marker)

replacement = '''    poster_high, poster_low = st.columns(2)

    with poster_high:
        st.error(
            """⚠️ Highest observed spill site in 2025

**Nearby town:** South Molton  
**Treatment site:** South Molton WWTW  
**Water company:** South West Water  
**Receiving water:** River Mole  
**2025 highest spill:** 261"""
        )

    with poster_low:
        st.success(
            """✅ Lowest observed spill site in 2025

**Nearby town:** Chester  
**Overflow site:** BATCHE TANKS CSO  
**Water company:** Dwr Cymru Welsh Water  
**Receiving water:** Bache Brook  
**2025 lowest spill:** 0"""
        )

    st.caption(
        "Poster-aligned dissertation results. The company ranking uses the audited "
        "counted-spill totals already built into this dashboard."
    )
'''

path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print("Fixed poster result card strings.")
