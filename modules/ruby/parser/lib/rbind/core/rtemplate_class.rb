
module Rbind
    class RTemplateParameter < RDataType
        def template?
            true
        end
    end

    class RTemplateClassSpecialization < RClass
        attr_accessor :template,:template_parameters

        def initialize(name,template,*parameters)
            super(name)
            @name = @name.gsub(">>","> >") # force overwrite to match c++ syntax
            @template = template
            @template_parameteres = parameters
        end

        def specialize_ruby
            template.specialize_ruby_specialization(self)
        end
    end

    class RTemplateClass < RClass
        extend ::Rbind::Logger

        # template parameters in the right order
        attr_reader :template_parameters

        def initialize(name,*parent_classes)
            raise "parent classes for template classes are not supported!" if !parent_classes.empty?
            @template_parameters =[]
            super
        end

        def template?
            true
        end

        # called by RNamespace
        def do_specialize(name,*parameters)
            klass = RTemplateClassSpecialization.new(name,self,*parameters)
            specialize(klass,*parameters)
        end

        # hook for implementing the specialization
        def specialize(klass,*parameters)
            RTemplateClass.log.info "RTemplateClass: #{name} default specialized with #{klass} -- specialization parameters: #{parameters}, template parameters: #{template_parameters}"
            # by default create a dummy implementation without additional arguments
            if parameters.size != @template_parameters.size
                raise ArgumentError, "RTemplateClass: template #{name} expects #{@template_parameters.size} parameters"
            end

            operations.each do |ops|
                ops.each do |o|
                    begin
                        RTemplateClass.log.debug "RTemplateClass: #{name} handle operation: #{o.name}"
                        op = o.dup

                        if op.kind_of?(RGetter) or op.kind_of?(RSetter)
                            attr = RAttribute.new(op.attribute.name, resolve_type(parameters, op.attribute.type))
                            op = op.class.new(attr)
                            klass.add_operation(op)
                            next
                        else
                            op.return_type = resolve_type(parameters, op.return_type.name)
                        end

                        op.parameters.each do |p|
                            rtype = resolve_type(parameters, p.type.name)
                            RTemplateClass.log.debug "RTemplateClass: #{name} specialized with #{klass} -- resolved #{p.type.name} -> #{rtype}"
                            p.type = rtype
                        end

                        klass.add_operation(op)
                    rescue => e
                        RTemplateClass.log.warn "RTemplateClass: #{name} could not add parameter #{e} #{e.backtrace}"

                    end
                end
            end
            klass
        end

        def resolve_type(parameters, param_name)
            # This resolves
            resolved_type = type(param_name, false)
            if not resolved_type
                raise ArgumentError, "RTemplateClass: template #{name} could not resolve template parameter #{param_name}"
            end

            if index = @template_parameters.index(resolved_type)
                resolved_type = type(parameters[index])
            end

            resolved_type
        end

        # hook for generating additional ruby code going to be embedded into the
        # class definition
        def specialize_ruby_specialization(klass)
        end

        def add_type(klass)
            if klass.kind_of?(RTemplateParameter)
                @template_parameters << klass
            end
            super
        end
    end
end
