-- this is a lua script for use in conky
--
-- bg_color=0x3b3b3b
-- bg_alpha=0.2
--
setting_table = {
    {
        name='fs_used_perc',
        arg='/',
        bg_color=0x012a2b,
        bg_alpha=0.1,
--        fg_colour=0x165cc4,
        fg_color=0xc35822,
        fg_alpha=0.4,
        start_x=185, start_y=523,
        length = 170,
        thickness=5
    },
    {
        name='fs_used_perc',
        arg='/home',
        bg_color=0x012a2b,
        bg_alpha=0.1,
--        fg_colour=0x165cc4,
        fg_color=0xc35822,
        fg_alpha=0.4,
        start_x=185, start_y=562,
        length=170,
        thickness=5
    },
}

require 'cairo'

function rgb_to_r_g_b(color, alpha)
    return ((color / 0x10000) % 0x100) / 255., ((color / 0x100) % 0x100) / 255., (color % 0x100) / 255., alpha
end

function draw_stats()
    if conky_window==nil then return end

    local cs=cairo_xlib_surface_create(conky_window.display, conky_window.drawable, conky_window.visual, conky_window.width, conky_window.height)
    local cr=cairo_create(cs)

    line_cap=CAIRO_LINE_CAP_BUTT

    for i, element in pairs(setting_table) do

        str = string.format('${%s %s}', element['name'], element['arg'])
        system_used_perc = tonumber(conky_parse(str)) / 100 * tonumber(element['length'])


        cairo_set_line_width (cr,element['thickness'])
        cairo_set_line_cap  (cr, line_cap)
        cairo_set_source_rgba (cr,rgb_to_r_g_b(element['bg_color'], element['bg_alpha']))
        cairo_move_to (cr,element['start_x'],element['start_y'])
        cairo_line_to (cr,element['start_x']+tonumber(element['length']), element['start_y'])
        cairo_stroke (cr)

        cairo_set_source_rgba (cr,rgb_to_r_g_b(element['fg_color'], element['fg_alpha']))
        cairo_move_to (cr,element['start_x'],element['start_y'])
        cairo_line_to (cr,element['start_x']+system_used_perc, element['start_y'])
        cairo_stroke (cr)
    end
end

function conky_main()
    draw_stats()
end
