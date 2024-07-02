-- this is a lua script for use in conky
--
-- bg_color=0x3b3b3b
-- bg_alpha=0.2
--
setting_table = {
    {
        name='fs_used_perc',
        arg='/',
        bg_color=0x3b3b3b,
        bg_alpha=0.4,
--        fg_colour=0x165cc4,
        fg_color=0xc35822,
        fg_alpha=0.4,
        start_x=185, start_y=670,
        length = 170,
        thickness=8
    },
    {
        name='fs_used_perc',
        arg='/home',
        bg_color=0x3b3b3b,
        bg_alpha=0.4,
--        fg_colour=0x165cc4,
        fg_color=0xc35822,
        fg_alpha=0.4,
        start_x=185, start_y=737,
        length=170,
        thickness=8
    },
}

require 'cairo'

function rgb_to_r_g_b(color, alpha)
    return ((color / 0x10000) % 0x100) / 255., ((color / 0x100) % 0x100) / 255., (color % 0x100) / 255., alpha
end

function conky_main()
end
