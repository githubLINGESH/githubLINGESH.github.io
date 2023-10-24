    import { ClerkProvider } from '../../node_modules@clerk/nextjs';

    export const ClerkProviderWrapper = ({ children }) => {
    return (
        <ClerkProvider publicApiKey={process.env.NEXT_PUBLIC_CLERK_PUBLIC_API_KEY}>
        {children}
        </ClerkProvider>
    );
    };
