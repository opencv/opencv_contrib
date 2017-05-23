require 'open3'

module Rbind
    class Rbind
        attr_reader :parser
        attr_reader :generator_c
        attr_reader :generator_ruby
        attr_accessor :includes
        attr_accessor :name
        attr_accessor :pkg_config
        attr_accessor :rbind_pkgs    # extern rbind pkgs

        def self.pkg_paths(pkg_name)
            out = IO.popen("pkg-config --cflags-only-I #{pkg_name}")
            paths = out.read.split("-I").delete_if(&:empty?).map do |i|
                i.gsub("\n","").gsub(" ","")
            end
            raise "Cannot find pkg paths for #{pkg_name}" if paths.empty?
            paths
        end

        def self.rbind_pkg_path(name)
            files = Gem.find_files("#{name}/rbind/extern.rbind")
            raise "Cannot find paths for rbind package #{name}" if files.empty?
            File.dirname(files.first)
        end

        def initialize(name,parser = DefaultParser.new)
            @name = GeneratorRuby.name
            @includes = []
            @pkg_config = []
            @rbind_pkgs= []
            @parser = parser

            lib_name = "rbind_#{name.downcase}"
            @generator_c = GeneratorC.new(@parser,name,lib_name)
            @generator_ruby = GeneratorRuby.new(@parser,name,lib_name)
            @generator_extern = GeneratorExtern.new(@parser)
        end

        def parse(*files)
            files.flatten.each do |path|
                raise ArgumentError, "File '#{path}' does not exist" if not File.exists?(path)
                parser.parse File.new(path).read
            end
        end

        def check_python
            out = IO.popen("which python")
            if(out.read.empty?)
                raise 'Cannot find python interpreter needed for parsing header files'
            end
            in_,out,err = Open3.popen3("python --version")
            str = err.read
            str = if str.empty?
                      out.read
                  else
                      str
                  end
            if(str =~ /[a-zA-Z]* (.*)/)
                if $1.to_f < 2.7
                    raise "Wrong python version #{$1}. At least python 2.7 is needed for parsing header files"
                end
            else
                raise 'Cannot determine python version needed for parsing header files'
            end
        end

        # parses other rbind packages
        def parse_extern
            # extern package are always paresed with the default parser 
            local_parser = DefaultParser.new(parser)
            @rbind_pkgs.each do |pkg|
                path = Rbind.rbind_pkg_path(pkg)
                config = YAML.load(File.open(File.join(path,"config.rbind")).read)
                path = File.join(path,"extern.rbind")
                ::Rbind.log.info "parsing extern rbind file #{path}"
                local_parser.parse(File.open(path).read,config.ruby_module_name)
            end
            self
        end

        def parse_headers_dry(*headers)
            check_python
            headers = if headers.empty?
                          includes
                      else
                          headers
                      end
            headers = headers.map do |h|
                "\"#{h}\""
            end
            path = File.join(File.dirname(__FILE__),'tools','hdr_parser.py')
            out = IO.popen("python #{path} #{headers.join(" ")}")
            out.read
        end

        def parse_headers(*headers)
            parser.parse parse_headers_dry(*headers)
        end

        def parse_header(header)
            parser.parse parse_headers_dry(header)
        end

        def build
            ::Rbind.log.info "build c wrappers"
            path = File.join(generator_c.output_path,"build")
            FileUtils.mkdir_p(path) if path && !File.directory?(path)
            Dir.chdir(path) do
                if !system("cmake -C ..")
                    raise "CMake Configure Error"
                end
                if !system("make")
                    raise "Make Build Error"
                end
            end
            if !system("cp #{File.join(path,"lib*.*")} #{generator_ruby.output_path}")
                raise "cannot copy library to #{generator_ruby.output_path}"
            end
            ::Rbind.log.info "all done !"
        end

        def generate(c_path = "src",ruby_path = "ruby/lib/#{name.downcase}")
            generate_c c_path,ruby_path
            generate_extern ruby_path,c_path
            generate_ruby ruby_path
        end

        def generate_ruby(path)
            ::Rbind.log.info "generate ruby ffi wrappers"
            @generator_ruby.required_module_names = rbind_pkgs
            @generator_ruby.generate(path)
        end

        def generate_c(path,ruby_path)
            ::Rbind.log.info "generate c wrappers"
            @generator_c.includes += includes
            @generator_c.includes.uniq!
            @generator_c.pkg_config = pkg_config
            @generator_c.generate(path,ruby_path)
        end

        def generate_extern(path,cpath)
            @generator_extern.generate(File.join(path,"rbind"),@generator_ruby.module_name,@generator_ruby.file_prefix,cpath)
        end

        def use_namespace(name)
            t = if name.is_a? String
                    parser.type(name)
                else
                    name
                end
            parser.use_namespace t
        end

        def type(*args)
            parser.type(*args)
        end

        def on_type_not_found(&block)
            @parser.on_type_not_found(&block)
        end

        def libs
            @generator_c.libs
        end

        def add_std_string(with_c_string = true)
            @generator_c.includes << "<string>"
            @generator_c.includes << "<cstring>" if with_c_string
            @parser.add_type(StdString.new("std::string",@parser))
            @parser.type_alias["basic_string"] = @parser.std.string
            self
        end

        def add_std_vector
            @generator_c.includes << "<vector>"
            @parser.add_type(StdVector.new("std::vector"))
            self
        end

        def add_std_map
            @generator_c.includes << "<map>"
            @parser.add_type(StdMap.new("std::map"))
        end

        def add_std_except
            @generator_c.includes << "<stdexcept>"
            exception = RClass.new(RBase.normalize("std::exception"))
            @parser.add_type(exception)

            runtime_error = RClass.new(RBase.normalize("std::runtime_error"))
            runtime_error.add_parent(exception)
            @parser.add_type(runtime_error)
        end

        def add_std_types
            add_std_vector
            add_std_string
            add_std_map
            add_std_except
        end

        def method_missing(m,*args)
            t = @parser.type(m.to_s,false,false)
            return t if t

            op = @parser.operation(m.to_s,false)
            return op if op

            super
        end
    end
end
