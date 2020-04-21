## 73b2にある物体一覧

```
$ (send-all (send *room73b2* :objects) :name) 
("room73b2-73b2-ground" "room73b2-locker2" "room73b2-locker1" "room73b2-door-left" "room73b2-door-right" "room73b2-foldable-desk" "room73b2-gifuplastic-900-cart" "room73b2-kirin-steppy-black-ladder" "room73b2-broom" "room73b2-bamboo-broom" "room73b2-azuma-broom" "room73b2-azuma-short-broom" "room73b2-coe-800-shelf" "room73b2-coe-450-shelf" "room73b2-uchida-shelf-1100" "room73b2-uchida-shelf-1300" "room73b2-uchida-shelf-1300" "room73b2-uchida-shelf-1300" "room73b2-uchida-shelf-1300" "room73b2-uchida-shelf-1300" "room73b2-uchida-shelf-1300" "room73b2-bariera-1200-right" "room73b2-bariera-1200-middle3-0" "room73b2-bariera-1200-middle3-1" "room73b2-bariera-1200-middle2" "room73b2-bariera-1200-middle-0" "room73b2-bariera-1200-middle-1" "room73b2-bariera-1200-middle-2" "room73b2-bariera-1200-corner" "room73b2-wanda" "room73b2-iemon" "room73b2-georgia-emerald-mountain" "room73b2-hitachi-fiesta-refrigerator" "room73b2-chair1" "room73b2-chair0" "room73b2-empty-box" "room73b2-bottle" "room73b2-sushi-cup2" "room73b2-trashbox0" "room73b2-karimoku-table" "room73b2-hrp2-parts-drawer" "room73b2-plus-590-locker" "room73b2-sharp-52-aquostv" "room73b2-askul-1200x700-desk-0" "room73b2-askul-1200x700-desk-1" "room73b2-askul-1200x700-desk-2" "room73b2-askul-1200x700-desk-3" "room73b2-askul-1200x700-desk-4" "room73b2-askul-1200x700-desk-5" "room73b2-askul-1200x700-desk-6" "room73b2-uchida-shelf-1100" "room73b2-askul-1000x700-desk" "room73b2-uchida-shelf-1300" "room73b2-askul-1200x700-desk-7" "room73b2-cupboard-right" "room73b2-cupboard-left" "room73b2-toshiba-clacio-refrigerator" "room73b2-bariera-1400-middle" "room73b2-bariera-900-middle-0" "room73b2-bariera-900-middle-1" "room73b2-bariera-900-left" "room73b2-askul-1400-desk" "room73b2-desk-0" "room73b2-desk-1" "room73b2-desk-2" "room73b2-unknown-1200-desk-0" "room73b2-unknown-1200-desk-1" "room73b2-unknown-1200-desk-2" "room73b2-unknown-1200-desk-3" "room73b2-unknown-1200-desk-4" "room73b2-external-wall-4" "room73b2-external-wall-3" "room73b2-external-wall-2" "room73b2-external-wall-1" "room73b2-external-wall-0" "room73b2-panelwall-0" "room73b2-sushi-cup" "room73b2-mug-cup" "room73b2-tray" "room73b2-kettle" "room73b2-knife" "room73b2-sponge" "room73b2-cup" "room73b2-dish" "room73b2-kitchen-shelf" "room73b2-kitchen" "/eng2/7f/room73B2-front-of-kitchenboard" "/eng2/7f/room73B2-sink-front" "/eng2/7f/room73B2-beside-chair" "/eng2/7f/room73B2-far-chair-back" "/eng2/7f/room73B2-tmp-chair-back" "/eng2/7f/room73B2-chair-back" "/eng2/7f/room73B2-table-front" "/eng2/7f/room73B2-table-back" "/eng2/7f/room73B2-table-side" "/eng2/7f/room73B2-front-kitchen-table" "/eng2/7f/room73B2-front-of-tv" "door-spot" "coe-spot" "broom-spot" "table-spot" "cook-spot" "fridge-front-spot" "init-spot")
```

## 部屋内の代表的な座標
```
$ (send *room73b2* :spots)
(#<cascaded-coords #X5620b8ad48d0 /eng2/7f/room73B2-front-of-kitchenboard  1350.0 1850.0 0.0 / 3.142 0.0 0.0> #<cascaded-coords #X5620b8ad45a0 /eng2/7f/room73B2-sink-front  1355.0 2450.0 0.0 / 3.142 0.0 0.0> #<cascaded-coords #X5620b8ad4528 /eng2/7f/room73B2-beside-chair  3860.0 350.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad4468 /eng2/7f/room73B2-far-chair-back  3470.0 1150.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad4270 /eng2/7f/room73B2-tmp-chair-back  3280.0 2150.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad41b0 /eng2/7f/room73B2-chair-back  3580.0 1150.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad3f88 /eng2/7f/room73B2-table-front  4200.0 1000.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad3c10 /eng2/7f/room73B2-table-back  5155.0 10.0 0.0 / 1.571 0.0 0.0> #<cascaded-coords #X5620b8ad3730 /eng2/7f/room73B2-table-side  5150.0 2180.0 0.0 / -1.571 0.0 0.0> #<cascaded-coords #X5620b8ad35b0 /eng2/7f/room73B2-front-kitchen-table  2293.0 1983.0 0.0 / 3.142 0.0 0.0> #<cascaded-coords #X5620b8ad3430 /eng2/7f/room73B2-front-of-tv  3700.0 1700.0 0.0 / 1.571 0.0 0.0> #<cascaded-coords #X5620b8ad32f8 door-spot  675.0 210.0 0.0 / 3.142 0.0 0.0> #<cascaded-coords #X5620b8ad31c0 coe-spot  1200.0 2300.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad3100 broom-spot  2250.0 1000.0 0.0 / -1.571 0.0 0.0> #<cascaded-coords #X5620b8ad2f50 table-spot  4100.0 1600.0 0.0 / 1.571 0.0 0.0> #<cascaded-coords #X5620b8ad2ed8 cook-spot  1100.0 1600.0 0.0 / 3.142 0.0 0.0> #<cascaded-coords #X5620b8ad1648 fridge-front-spot  4800.0 1480.0 0.0 / 0.0 0.0 0.0> #<cascaded-coords #X5620b8ad1510 init-spot  500.0 0.0 0.0 / 0.0 0.0 0.0>)
```

移動先については
models/room73b2-scene.l
の下の方にある位置座標を参考にする。
例えば、cupboardの左側については、
```
 (send (room73b2-cupboard-left) :transform (make-coords :pos (float-vector 2048.0 3526.5 0.0) :rot #2f((2.220446e-16 1.0 0.0) (-1.0 2.220446e-16 0.0) (0.0 0.0 1.0))))
```
とあるが、これだと食器棚の中の座標になるので、もし移動先に使うなら手前側にしないと行けない。また、向きも180度逆になっているので回転させる。

```
(send *robot* :move-to  (send (room73b2-cupboard-left) :transform (make-coords :pos (float-vector 2048.0 (- 3526.5 550) 0.0) :rot #2f((2.220446e-16 -1.0 0.0) (1.0 2.220446e-16 0.0) (0.0 0.0 1.0)))) :world)

```

これで、食器棚の前に移動できる。