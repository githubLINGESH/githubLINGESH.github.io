    import { ClerkProviderWrapper } from './utils/clerk';

    function MyApp({ Component, pageProps }) {
    return (
        <ClerkProviderWrapper>
        <Component {...pageProps} />
        </ClerkProviderWrapper>
    );
    }

    export default MyApp;
