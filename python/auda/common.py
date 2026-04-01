class AllowCustomArgs:
    context_settings = {
        'allow_extra_args': True,
        'ignore_unknown_options': True,
    }

    @staticmethod
    def parse_kwargs(args: list[str]) -> dict[str, str]:
        context: dict[str, str] = {}
        for arg in args:
            if not arg.startswith('--'):
                raise ValueError(f'Invalid argument: {arg}')

            if '=' not in arg:
                key = arg[2:]
                context[key] = 'true'
            else:
                key, value = arg[2:].split('=', 1)
                context[key] = value

        return context
