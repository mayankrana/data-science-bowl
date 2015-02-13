require 'image'
require 'nn'

function random_v_flip(_im)
   local r = torch.rand(1)
   if r > 0.5 then
      _im = image.vflip(_im)
   end
   return _im
end

function random_h_flip(_im)
   local r = torch.rand(1)
   if r > 0.5 then
      _im = image.hflip(_im)
   end
   return _im
end

function random_rotation(_im)
   local r = torch.rand(1)
   if r > 0.5 then
      --if vertical or horizontal random flip is applied then rotation < 90
      _im = image.rotate(im, math.pi/2)
   end
   return _im
end

function scale(_im)
   --rescale it to sampleSize
   _im = image.scale(_im, sampleSize[2], sampleSize[3])
   return _im
end

--Crops square image of size = image's smaller dimension
--Randomly select the position or crop along the longer dimension
function random_crop(_im)
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
--   print(im_size)
--   print(crop_size)
--   print(start_x)
--   print(start_y)
--   print(end_x)
--   print(end_y)

   _im = image.crop(_im, start_x, start_y, end_x, end_y)

--   print(_im:size())
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


--TODO
function jitter(_im)
   local max_jitter = 0.15
   local _r = torch.rand(3)
   local im_size = _im[1]:size()

   local crop_size = {math.floor(torch.min(_im_size) * (1-max_jitter)),
                       math.floor(torch.min(_im_size) * (1-max_jitter))}

   if _r[1] > 0.5 then
      local scale = torch.uniform(0.9,1.1)
   end
   local start_x = math.ceil(_r[1] * (im_size[1] - crop_size[1] - 1))
   local start_y = math.ceil(_r[2] * (im_size[2] - crop_size[2] - 1))
   local end_x = start_x + crop_size[1]
   local end_y = start_y + crop_size[2]
   _im = image.crop(_im, start_x, start_y, end_x, end_y)
   _im = scale(_im)
   return _im
end

function dataAugmentation(_im)
   _im = random_crop(_im)
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
