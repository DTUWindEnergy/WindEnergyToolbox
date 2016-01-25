
function [time, data, info] = gtsdf_load(filename)

%  gtsdf_load Load a General time series data format - file
%   Example:
%       [time, data, info] = gtsdf_load('tests/test_files/test.hdf5')
%

    
    if nargin==0
        filename = 'examples/all.hdf5';
    end

    function value = att_value(name, addr, default)
        try
            value = h5readatt(filename, addr,name);
        catch
            if nargin==3
                value = default;
            else
                value = '';
            end
        end
    end

    function r = read_dataset(name,  addr, default)
        try
            r = h5read(filename, strcat(addr,name));
        catch
            r = default;
        end 
    end
    

    if not (strcmpi(lower(att_value('type','/')), 'general time series data format'))
        error('HDF5 file must contain a ''type''-attribute with the value ''General time series data format''')
    end
    if strcmp(att_value('no_blocks','/'),'')
        error('HDF5 file must contain an attribute named ''no_blocks''')
    end
    hdf5info = h5info(filename);
    if not (strcmp(hdf5info.Groups(1).Name,'/block0000'))
        error('HDF5 file must contain a group named ''block0000''')
    end

    datainfo = h5info(filename,'/block0000/data');
    no_attributes = datainfo.Dataspace.Size(1);
    type = att_value('type','/');
    name = att_value('name', '/','no_name');
    description = att_value('description', '/');
    
    attribute_names = read_dataset('attribute_names','/', {});
    attribute_units = read_dataset('attribute_units','/', {});
    attribute_descriptions = read_dataset('attribute_descriptions','/', {});
    
    
    info = struct('type',type, 'name', name, 'description', description, 'attribute_names', {attribute_names}, 'attribute_units', {attribute_units}, 'attribute_descriptions',{attribute_descriptions});
    
    no_blocks = att_value('no_blocks','/');
    time = [];
    data = [];
    for i=0:no_blocks-1
       blockname = num2str(i,'/block%04d/');
       blokdatainfo = h5info(filename,strcat(blockname,'data'));
       no_observations = datainfo.Dataspace.Size(2);
       blocktime = double(read_dataset('time', blockname, [0:no_observations-1]'));
       blocktime_start = double(att_value('time_start',blockname,0));
       blocktime_step = double(att_value('time_step',blockname,1));
       time = [time;(blocktime*blocktime_step) + blocktime_start];
       
       block_data = read_dataset('data', blockname)';
       if isinteger(block_data)
           nan_pos = block_data==intmax(class(block_data));
           block_data = double(block_data);
           block_data(nan_pos) = nan;
           gains = double(read_dataset('gains',blockname,1.));
           offsets = double(read_dataset('offsets', blockname,0));
           for c = 1:no_attributes
                block_data(:,c) = block_data(:,c)*gains(c)+offsets(c);
           end 
       end
       data = [data;block_data];
    end
end
