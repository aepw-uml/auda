# [AEPW x UML Data Analysis (AUDA)](https://github.com/aepw-uml/ada-server)

## How to refactor (last version)

- Global task names with protocols
- Remove all the "__common.py" modules
- Always use `np.ndarray` instead of nested lists
- Task specific inputs
- Use `step` instead of `task` for AUDA pipelines
- **Steps** should be more general and reusable
- 

```shell
auda pipe run DS-YEAR-PW:location=Japan
```
