# Model Context Protocol: Architecture and Security

The Model Context Protocol (MCP) is an open protocol for connecting AI applications to external tools and context providers through a consistent client-server interface. It helps separate an AI application's orchestration logic from the services that expose capabilities.

MCP standardizes communication; it does not make a connected server trustworthy or automatically authorize every action.

## Main components

- **Host:** The AI application that manages the user experience and coordinates model interactions.
- **Client:** A component inside the host that maintains a connection to an MCP server.
- **Server:** A process or service that exposes capabilities to clients.

Servers can expose several kinds of capability:

- **Tools:** Actions the client may invoke, such as searching records or creating a ticket.
- **Resources:** Context that can be read, such as files, database records, or service metadata.
- **Prompts:** Reusable prompt templates or workflows that a client can present to a model.

The host decides how discovered capabilities are presented and whether a request is allowed. A tool being advertised by a server is not the same as user authorization to run it.

## Protocol shape

MCP uses JSON-RPC messages for requests, responses, and notifications. A connection typically performs initialization and capability negotiation before normal operations.

Common transport choices include:
- **stdio:** Useful when a host launches a local server process and communicates over standard input and output.
- **Streamable HTTP:** Useful for remote server communication over HTTP with support for streaming behavior.

Transport and authentication details should follow the current MCP specification and the security requirements of the deployment.

## A typical interaction

1. The host establishes a client connection and initializes the protocol session.
2. The client discovers available tools, resources, or prompts.
3. The host decides which capabilities are appropriate for the current task.
4. The model may request a tool call using the advertised schema.
5. The client sends the request to the server.
6. The host treats the result as external data, validates it, and decides what to do next.

The server should enforce authorization independently. Do not rely on the model to enforce permissions.

## Security boundaries

### Least privilege

Give each server only the credentials and resource access it needs. Separate read-only capabilities from write or destructive actions. Avoid broad filesystem, shell, or database access when a narrow API is sufficient.

### Input validation

Validate tool arguments on the server:
- enforce types, ranges, and required fields;
- validate resource identifiers and tenant ownership;
- reject unexpected fields where strict schemas are appropriate;
- use parameterized database queries;
- enforce timeouts and request-size limits.

A JSON schema validates structure, not business authorization.

### Untrusted content

Tool results, resource text, and external documents can contain prompt injection or misleading instructions. Treat them as data, not as higher-priority instructions. Keep policy and authorization decisions outside retrieved content.

### Network and secret protection

For remote servers:
- authenticate clients and authorize each operation;
- validate origin and host expectations;
- protect credentials and rotate them;
- defend against server-side request forgery and unsafe URL fetching;
- use secure transport and avoid logging secrets or sensitive payloads.

### Human approval

Require explicit approval for consequential side effects such as sending messages, changing access, deleting records, spending money, or modifying production systems. Show the action and its parameters before execution.

## Operational checklist

1. Inventory servers and the capabilities each one exposes.
2. Pin and review server versions and dependencies.
3. Authenticate remote connections and scope credentials.
4. Validate every tool call server-side.
5. Log action metadata with sensitive fields redacted.
6. Rate-limit expensive or risky operations.
7. Provide cancellation, timeout, and failure handling.
8. Test prompt injection, permission boundaries, and cross-tenant access.
9. Revoke access promptly when a server or credential is no longer needed.

## Practice

Design an MCP server that exposes read-only customer lookup and a separate refund-creation tool. Define schemas, tenant checks, approval requirements, audit fields, and rate limits. Test a request that attempts to read another tenant's records and one that asks the model to issue an unapproved refund.

## References

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/)
- [MCP security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)

**Key takeaway:** MCP standardizes how AI applications discover and invoke external capabilities. Secure systems still require least privilege, server-side authorization, untrusted-output handling, and explicit approval for consequential actions.