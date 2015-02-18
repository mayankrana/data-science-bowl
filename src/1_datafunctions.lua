require 'image'
require 'nn'

function random_v_flip(_im)
   local _r = torch.rand(1)
   if _r[1] > 0.5 then
      _im = image.vflip(_im)
   end
   return _im
end

function random_h_flip(_im)
   local _r = torch.rand(1)
   if _r[1] > 0.5 then
      _im = image.hflip(_im)
   end
   return _im
end

function random_rotation(_im)
   local _r = torch.rand(2)
   if _r[1] > 0.5 then
      --if vertical or horizontal random flip is applied then rotation < 90
      --TODO Check if rotations non multiple of pi/2 helps???
      _im = image.rotate(_im, math.pi * _r[2])
   end
   return _im
end

function scaleToSampleSize(_im)
   --rescale it to sampleSize
   _im = image.scale(_im, sampleSize[2], sampleSize[3])
   return _im
end

--ZOOM IN + Translational jitter effect
--Randomly select the position and crops the image according to scale
function random_crop(_im)
   local max_crop = 0.2--max % area which can cropped and removed
   local _scale = torch.uniform(1-max_crop, 1)
   local new_size = {math.floor(_im[1]:size(1) * _scale),
                     math.floor(_im[1]:size(2) * _scale)}

   local _r = torch.rand(2)

   local start_x = math.ceil(_r[1] * (_im[1]:size(2) - new_size[2]))
   local start_y = math.ceil(_r[2] * (_im[1]:size(1) - new_size[1]))
   local end_x = startX + new_size[2]
   local end_y = startY + new_size[1]

   _im = image.crop(_im, start_x, start_y, end_x, end_y)

   return _im
end


--Crops square image of size = image's smaller dimension
--Randomly select the position or crop along the longer dimension
--Preservs the shorter dimension and aspect ratio
function random_square_crop(_im)
   local im_size = torch.Tensor(2):zero()
   im_size[1] = _im[1]:size(1)
   im_size[2] = _im[1]:size(2)
   local min_size = 0
   if(im_size[1] < im_size[2]) then
      min_size = im_size[1]
   else
      min_size = im_size[2]
   end
   local crop_size = torch.Tensor(2):fill(min_size)
   local start_x = 0
   local start_y = 0
   local end_x = crop_size[2]
   local end_y = crop_size[1]

   local _r = torch.rand(1)

   if(im_size[1] < im_size[2]) then
      start_x = math.ceil(_r[1] * (im_size[2]-crop_size[2]))
      end_x = start_x + crop_size[2]
   else
      start_y = math.ceil(_r[1] * (im_size[1]-crop_size[1]))
      end_y = start_y + crop_size[1]
   end

   _im = image.crop(_im, start_x, start_y, end_x, end_y)

   return _im
end

--Crops square image of size = image's smaller dimension
--Crops from the middle of the longer dimension
function middle_crop(_im)
   local im_size = torch.Tensor(2):zero()
   im_size[1] = _im[1]:size(1)
   im_size[2] = _im[1]:size(2)
   local min_size = 0
   if(im_size[1] < im_size[2]) then
      min_size = im_size[1]
   else
      min_size = im_size[2]
   end
   local crop_size = torch.Tensor(2):fill(min_size)
   local start_x = 1
   local start_y = 1
   local end_x = crop_size[2]
   local end_y = crop_size[1]

   --if landscape mode
   if(im_size[1] < im_size[2]) then
      start_x = math.ceil(im_size[2]/2 - crop_size[2]/2 + 1)
      end_x = start_x + crop_size[2] - 1
   else -- if portrait mode
      start_y = math.ceil(im_size[1]/2-crop_size[1]/2 + 1)
      end_y = start_y + crop_size[1] - 1
   end
   _im = image.crop(_im, start_x, start_y, end_x, end_y)

   return _im
end

function dataAugmentation(_im)
   --ZOOM IN + Translational
--   _im = random_crop(_im)
   --Square crop image preserving the shorter dimension and aspect ratio
   _im = random_square_crop(_im)
   _im = random_v_flip(_im)
   _im = random_h_flip(_im)
   _im = random_rotation(_im)
   return _im
end

function dataNormalization(_im)
   _im:add(-_im:mean())
   _im:div(_im:std())
   return _im
end
