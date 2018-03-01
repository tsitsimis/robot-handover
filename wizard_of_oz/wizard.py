from Tkinter import *
from button_callbacks import *


master = Tk()
master.title("NAO - Wizard of Oz")

btn_width = 16

#### GENERAL
col_general = 0
Label(master, text="GENERAL").grid(row=0, column=col_general)

# ip
ip = Entry(master, width=btn_width)
ip.insert(0, "192.168.0.121")
ip.grid(row=1, column=col_general)

# port
port = Entry(master, width=btn_width)
port.insert(0, "9559")
port.grid(row=2, column=col_general)

# stiffness on
stif_on_btn = Button(master, command=lambda: nao_stiffness(ip.get(), int(port.get()), 1.0))
stif_on_btn.config(text="Stiff ON", width=btn_width)
stif_on_btn.grid(row=3, column=col_general)

# stiffness off
stif_off_btn = Button(master, command=lambda: nao_stiffness(ip.get(), int(port.get()), 0.0))
stif_off_btn.config(text="Stiff OFF", width=btn_width)
stif_off_btn.grid(row=4, column=col_general)

# camera on
cam_on_btn = Button(master, command=lambda: nao_open_camera(ip.get(), int(port.get())))
cam_on_btn.config(text="Camera ON", width=btn_width, bg="blue")
cam_on_btn.grid(row=5, column=col_general)

# camera off
cam_off_btn = Button(master, command=lambda: nao_close_camera(ip.get(), int(port.get())))
cam_off_btn.config(text="Camera OFF", width=btn_width, bg="blue")
cam_off_btn.grid(row=6, column=col_general)

#### POSTURES
col_postures = 1
Label(master, text="POSTURES").grid(row=0, column=col_postures)

# Sit
sit_btn = Button(master, command=lambda: nao_posture(ip.get(), int(port.get()), "Sit"))
sit_btn.config(text="Sit", width=btn_width)
sit_btn.grid(row=1, column=col_postures)

# Stand
stand_btn = Button(master, command=lambda: nao_posture(ip.get(), int(port.get()), "Stand"))
stand_btn.config(text="Stand", width=btn_width)
stand_btn.grid(row=2, column=col_postures)

# Crouch
crouch_btn = Button(master, command=lambda: nao_posture(ip.get(), int(port.get()), "Crouch"))
crouch_btn.config(text="Crouch", width=btn_width)
crouch_btn.grid(row=3, column=col_postures)

# Bend
bend_btn = Button(master, command=lambda: nao_bend(ip.get(), int(port.get()), 1))
bend_btn.config(text="Bend front", width=btn_width)
bend_btn.grid(row=4, column=col_postures)

# Walk Forward
walk_dist = Entry(master, width=5)
walk_dist.insert(0, "0.2")
walk_dist.grid(row=5, column=col_postures, sticky="e")

walk_btn = Button(master, command=lambda: nao_walk(ip.get(), int(port.get()), float(walk_dist.get())))
walk_btn.config(text="Walk", width=btn_width/2, bg="red")
walk_btn.grid(row=5, column=col_postures, sticky="w")

# turn_angle = 10 * almath.TO_RAD
turn_btn = Button(master, command=lambda: nao_turn(ip.get(), int(port.get()), float(turn_angle.get())*almath.TO_RAD))
turn_btn.config(text="<", width=btn_width/5, bg="red")
turn_btn.grid(row=6, column=col_postures, sticky="w")

turn_btn = Button(master, command=lambda: nao_turn(ip.get(), int(port.get()), -float(turn_angle.get())*almath.TO_RAD))
turn_btn.config(text=">", width=btn_width/5, bg="red")
turn_btn.grid(row=6, column=col_postures, sticky="e")

turn_angle = Entry(master, width=5)
turn_angle.insert(0, "90")
turn_angle.grid(row=6, column=col_postures, sticky="n")

# Bend down
bend_down_btn = Button(master, command=lambda: nao_bend(ip.get(), int(port.get()), 2))
bend_down_btn.config(text="Bend down", width=btn_width)
bend_down_btn.grid(row=7, column=col_postures)


#### BODY PARTS
col_body = 2
Label(master, text="ARM + HEAD").grid(row=0, column=col_body)

# wrist turned
wrist_turn_btn = Button(master, command=lambda: nao_turn_wrist(ip.get(), int(port.get()), -57 * almath.TO_RAD))
wrist_turn_btn.config(text="Turn Wrist", width=btn_width, bg="blue")
wrist_turn_btn.grid(row=1, column=col_body)

# wrist default
wrist_turn_btn = Button(master, command=lambda: nao_turn_wrist(ip.get(), int(port.get()), 0 * almath.TO_RAD))
wrist_turn_btn.config(text="Wrist default", width=btn_width, bg="blue")
wrist_turn_btn.grid(row=2, column=col_body)

# move arm
move_arm_btn = Button(master, command=lambda: nao_move_arm(ip.get(), int(port.get())))
move_arm_btn.config(text="Move arm", width=btn_width, bg="green")
move_arm_btn.grid(row=3, column=col_body)

# un-move arm
unmove_arm_btn = Button(master, command=lambda: nao_unmove_arm(ip.get(), int(port.get())))
unmove_arm_btn.config(text="Un-move arm", width=btn_width, bg="green")
unmove_arm_btn.grid(row=4, column=col_body)

# lower arm
lower_arm_btn = Button(master, command=lambda: nao_lower_arm(ip.get(), int(port.get())))
lower_arm_btn.config(text="Lower arm", width=btn_width)
lower_arm_btn.grid(row=5, column=col_body)

# lift arm
lift_arm_btn = Button(master, command=lambda: nao_lift_arm(ip.get(), int(port.get())))
lift_arm_btn.config(text="Lift arm", width=btn_width)
lift_arm_btn.grid(row=6, column=col_body)

# open-close hand
open_close_hand_btn = Button(master, command=lambda: nao_close_open_hand(ip.get(), int(port.get()), int(entry_hand.get())))
open_close_hand_btn.config(text="Open-Close hand", width=btn_width-6, bg="pink")
open_close_hand_btn.grid(row=7, column=col_body, sticky="w")

entry_hand = Entry(master, width=5)
entry_hand.insert(0, "2")
entry_hand.grid(row=7, column=col_body, sticky="e")

# close hand
close_hand_btn = Button(master, command=lambda: nao_close_hand(ip.get(), int(port.get())))
close_hand_btn.config(text="Close hand", width=btn_width, bg="pink")
close_hand_btn.grid(row=8, column=col_body)

# open hand
open_hand_btn = Button(master, command=lambda: nao_open_hand(ip.get(), int(port.get())))
open_hand_btn.config(text="Open hand", width=btn_width, bg="pink")
open_hand_btn.grid(row=9, column=col_body, pady=(0, 20))

# head
Label(text="Head").grid(row=10, column=col_body)

head_left_btn = Button(master, command=lambda: nao_head_yaw(ip.get(), int(port.get()), yaw=40, pitch=-20))
head_left_btn.config(text="<", width=btn_width/5)
head_left_btn.grid(row=11, column=col_body, sticky="w")

head_right_btn = Button(master, command=lambda: nao_head_yaw(ip.get(), int(port.get()), yaw=-40, pitch=-20))
head_right_btn.config(text=">", width=btn_width/5)
head_right_btn.grid(row=11, column=col_body, sticky="e")

head_up_btn = Button(master, command=lambda: nao_head_yaw(ip.get(), int(port.get()), yaw=0, pitch=-30))
head_up_btn.config(text="/\\", width=btn_width/5)
head_up_btn.grid(row=11, column=col_body)

# head_down_btn = Button(master, command=lambda: nao_head_yaw(ip.get(), int(port.get())))
# head_down_btn.config(text="\/", width=btn_width/5)
# head_down_btn.grid(row=11, column=col_body)


# head + hand
# head_hand_btn = Button(master, command=lambda: nao_head_hand(ip.get(), int(port.get()),
#                                                              int(entry_head.get()), int(entry_hand.get())))
# head_hand_btn.config(text="Head + Hand", width=btn_width)
# head_hand_btn.grid(row=10, column=col_body)

# # lift arm point
# lift_arm_point_btn = Button(master, command=lambda: nao_lift_arm_point(ip.get(), int(port.get())))
# lift_arm_point_btn.config(text="Lift arm point", width=btn_width)
# lift_arm_point_btn.grid(row=10, column=col_body)

#### VERBAL
col_verbal = 3
Label(master, text="VERBAL").grid(row=0, column=col_verbal)

# geia sou
# geia_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "hello"))
# geia_btn.config(text="geia sou", width=btn_width)
# geia_btn.grid(row=1, column=col_verbal)

# mporeis na me voithiseis
help_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "help"))           # girl
# help_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "Voithiseis"))   # boy
help_btn.config(text="mporeis na me voithiseis", width=btn_width)
help_btn.grid(row=2, column=col_verbal)

# dwse mou
dwsemou_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "dwsmounatopiaso"))     # girl
# dwsemou_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "DoseMou"))           # boy
dwsemou_btn.config(text="dwse mou", width=btn_width)
dwsemou_btn.grid(row=3, column=col_verbal)

# den to ftanw. dwse mou
denftanw_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "dentoftano"))     # girl
# denftanw_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "DenFtano"))     # boy
denftanw_btn.config(text="den ftanw. dwse mou", width=btn_width)
denftanw_btn.grid(row=4, column=col_verbal)

# pio konta
# piokonta_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "piokonta"))
# piokonta_btn.config(text="pio konta", width=btn_width)
# piokonta_btn.grid(row=5, column=col_verbal)

# thanks
thanks_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "thanks"))           # girl
# thanks_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "Efxaristo"))      # boy
thanks_btn.config(text="Euxaristw polu", width=btn_width)
thanks_btn.grid(row=6, column=col_verbal)

# nai ayto
# naiayto_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "naiauto"))
# naiayto_btn.config(text="nai ayto", width=btn_width)
# naiayto_btn.grid(row=7, column=col_verbal)

# oxi ayto
# oxiayto_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "oxiauto"))
# oxiayto_btn.config(text="oxi ayto", width=btn_width)
# oxiayto_btn.grid(row=8, column=col_verbal)

# epomeno
epomeno_btn = Button(master, command=lambda: nao_say(ip.get(), int(port.get()), "epomeno_paixnidi"))
epomeno_btn.config(text="epomeno", width=btn_width)
epomeno_btn.grid(row=9, column=col_verbal)

# on-load
# turn on camera
# try:
#     nao_open_camera(ip.get(), int(port.get()))
#     print "camera on"
# except:
#     print "camera off"


w, h = master.winfo_screenwidth(), master.winfo_screenheight()
master.geometry("%dx%d+0+0" % (700, 500))
master.mainloop()
quit()
