require 'yaml'

module Rbind
    class GeneratorExtern
        Config = Struct.new(:ruby_module_name,:file_prefix,:cpath)

        attr_accessor :output_path
        attr_accessor :ruby_module_name
        attr_accessor :file_prefix
        attr_accessor :cpath

        def self.normalize_type_name(name)
            name.gsub('::','.').gsub(" ","")
        end

        def self.normalize_operation_name(name)
            name.gsub('::','.').gsub(" ","")
        end

        def initialize(root)
            @root = root
        end

        def generate(path = @output_path,ruby_module_name = @ruby_module_name,file_prefix = @file_prefix,cpath=@cpath)
            @output_path = path
            @ruby_module_name = ruby_module_name
            @file_prefix = file_prefix
            @cpath = File.expand_path(cpath)
            FileUtils.mkdir_p(path) if path && !File.directory?(path)
            file_extern = File.new(File.join(path,"extern.rbind"),"w")
            file_config = File.new(File.join(path,"config.rbind"),"w")

            #write all types so they get parsed first
            @root.each_type do |t|
                if t.is_a? RClass
                    file_extern.write "class #{GeneratorExtern.normalize_type_name(t.full_name)}\n"
                end
            end

            # write all consts
            @root.each_const do |c|
                file_extern.write "const #{GeneratorExtern.normalize_type_name(c.full_name)}\n"
            end

            #write all operations
            @root.each_type do |t|
                if t.is_a? RClass
                    t.each_operation do |op|
                        r = if op.return_type
                                GeneratorExtern.normalize_type_name(op.return_type.full_name)
                            end
                        file_extern.write "#{GeneratorExtern.normalize_operation_name(op.full_name)} #{r}\n"
                        op.parameters.each do |p|
                            file_extern.write "   #{GeneratorExtern.normalize_type_name(p.type.full_name)} #{p.name}"\
                                              "#{" #{p.default_value}" if p.default_value}#{" /O" unless p.const?}\n"
                        end
                    end
                end
            end

            file_extern.write("\n")
            file_config.write Config.new(ruby_module_name,file_prefix,@cpath).to_yaml
        end
    end
end
