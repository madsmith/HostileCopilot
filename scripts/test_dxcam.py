import dxcam
camera = dxcam.create()

def grab_screenshot_and_display(device_idx, output_idx):
    camera = dxcam.create(device_idx=device_idx, output_idx=output_idx)
    frame = camera.grab()
    print(frame.shape)
    from PIL import Image
    Image.fromarray(frame).show()

print(f"Devices: {dxcam.device_info()}")
print(f"Outputs: {dxcam.output_info()}")
grab_screenshot_and_display(0, 0)
grab_screenshot_and_display(0, 1)
