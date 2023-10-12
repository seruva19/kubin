(() => {
  kubin.visualizeAddedCondition('.t2i_params .kd-networks .label-wrap span', '.t2i_params .networks-enable-lora [type=checkbox]', false)
  kubin.visualizeAddedCondition('.i2i_params .kd-networks .label-wrap span', '.i2i_params .networks-enable-lora [type=checkbox]', false)
  kubin.visualizeAddedCondition('.mix_params .kd-networks .label-wrap span', '.mix_params .networks-enable-lora [type=checkbox]', false)
  kubin.visualizeAddedCondition('.inpaint_params .kd-networks .label-wrap span', '.inpaint_params .networks-enable-lora [type=checkbox]', false)
  kubin.visualizeAddedCondition('.outpaint_params .kd-networks .label-wrap span', '.outpaint_params .networks-enable-lora [type=checkbox]', false)
})()