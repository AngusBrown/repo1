function imagesAltered=get_altered_images(images)

imagesAltered=[];

for n=1:size(images,3)

image=images(:,:,n);

%% Rotate
step=0;
for theta=-16:8:16
    step=step+1;
    imageRotate{step}=imrotate(image,theta, 'crop', 'bilinear');
    
end

%% Warp and Crop
centerO=floor(size(image)/2+0.5);
step=0;
shear=0.3;
for xsk=-shear:shear:shear
    for ysk=-shear:shear:shear
        if xsk~=ysk
            step=step+1;
            tform = affine2d([1,xsk,0;ysk,1,0;0,0,1]);
            imageWarp=imwarp(image,tform);
            center=floor(size(imageWarp)/2+0.5);
            
            %x
%             if rem(size(imageWarp,1)/2+0.5,1)<0.00001 %if all are odd
%                 cropValuesX=center(1)-centerO(1):center(1)+centerO(1);
%             else
                cropValuesX=center(1)-centerO(1)+1:center(1)+centerO(1);
%             end
            
            %y
%             if rem(size(imageWarp,2)/2+0.5,1)<0.00001 %if all are odd
%                 cropValuesY=center(2)-centerO(2):center(2)+centerO(2);
%             else
                cropValuesY=center(2)-centerO(2)+1:center(2)+centerO(2);
%             end
            imageWarpCrop{step}=imageWarp(cropValuesX(1,:), cropValuesY(1,:));
        end
    end
end


imageAltered=[imageRotate, imageWarpCrop];

imagesAltered=cat(3,imagesAltered, cat(3,imageAltered{:}));

end









